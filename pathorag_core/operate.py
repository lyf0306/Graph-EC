# operate.py
import asyncio
import math
import json
import re
import jieba
from rank_bm25 import BM25Okapi
import torch
import numpy as np
from tqdm.asyncio import tqdm as tqdm_async
from typing import Union
from collections import Counter, defaultdict
import warnings

from .instrumentation import traced, trace_span
from .utils import (
    logger,
    clean_str,
    compute_mdhash_id,
    decode_tokens_by_tiktoken,
    encode_string_by_tiktoken,
    is_float_regex,
    list_of_list_to_csv,
    pack_user_ass_to_openai_messages,
    split_string_by_multi_markers,
    truncate_list_by_token_size,
    process_combine_contexts,
    compute_args_hash,
    handle_cache,
    save_to_cache,
    CacheData,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    TextChunkSchema,
    QueryParam,
)
from .prompt import GRAPH_FIELD_SEP, PROMPTS
from .hyper_attention import GLOBAL_ENTITY_CACHE, compute_hyperedge_scores_sync

_READ_SEM = asyncio.Semaphore(30)

async def _gather_reads(tasks: list):
    """对 Neo4j 读操作批量并发，内部用 Semaphore(30) 限流防止连接池耗尽。"""
    async def _wrap(coro):
        async with _READ_SEM:
            return await coro
    return await asyncio.gather(*[_wrap(t) for t in tasks])


def chunking_by_token_size(
    content: str, overlap_token_size=128, max_token_size=1024, tiktoken_model="gpt-4o"
):
    tokens = encode_string_by_tiktoken(content, model_name=tiktoken_model)
    results = []
    for index, start in enumerate(
        range(0, len(tokens), max_token_size - overlap_token_size)
    ):
        chunk_content = decode_tokens_by_tiktoken(
            tokens[start : start + max_token_size], model_name=tiktoken_model
        )
        results.append(
            {
                "tokens": min(max_token_size, len(tokens) - start),
                "content": chunk_content.strip(),
                "chunk_order_index": index,
            }
        )
    return results


async def _handle_entity_relation_summary(
    entity_or_relation_name: str,
    description: str,
    global_config: dict,
) -> str:
    use_llm_func: callable = global_config["llm_model_func"]
    llm_max_tokens = global_config["llm_model_max_token_size"]
    tiktoken_model_name = global_config["tiktoken_model_name"]
    summary_max_tokens = global_config["entity_summary_to_max_tokens"]
    language = global_config["addon_params"].get(
        "language", PROMPTS["DEFAULT_LANGUAGE"]
    )

    tokens = encode_string_by_tiktoken(description, model_name=tiktoken_model_name)
    if len(tokens) < summary_max_tokens: 
        return description
    prompt_template = PROMPTS["summarize_entity_descriptions"]
    use_description = decode_tokens_by_tiktoken(
        tokens[:llm_max_tokens], model_name=tiktoken_model_name
    )
    context_base = dict(
        entity_name=entity_or_relation_name,
        description_list=use_description.split(GRAPH_FIELD_SEP),
        language=language,
    )
    use_prompt = prompt_template.format(**context_base)
    logger.debug(f"Trigger summary: {entity_or_relation_name}")
    summary = await use_llm_func(use_prompt, max_tokens=summary_max_tokens)
    return summary


async def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
    now_hyper_relation: str,
):
    if len(record_attributes) < 6 or record_attributes[0] != '"entity"' or now_hyper_relation == "":
        return None
        
    entity_name = clean_str(record_attributes[1].upper())
    if not entity_name.strip():
        return None
    entity_type = clean_str(record_attributes[2].upper())
    entity_description = clean_str(record_attributes[3])
    
    raw_role = clean_str(record_attributes[4].upper()).replace('"', '').replace("'", "").strip()
    
    # 角色白名单映射
    ROLE_MAPPING = {
        "CONDITION": "CONDITION",
        "INPUT": "CONDITION",
        "PREREQUISITE": "CONDITION",
        "RECOMMENDATION": "RECOMMENDATION",
        "OUTCOME": "RECOMMENDATION",
        "ACTION": "RECOMMENDATION",
        "TREATMENT": "RECOMMENDATION",
        "CONTEXT": "CONTEXT",
        "BACKGROUND": "CONTEXT",
        "EVIDENCE": "EVIDENCE",
        "SOURCE": "EVIDENCE"
    }
    
    edge_role = ROLE_MAPPING.get(raw_role, "CONDITION")
    
    weight_str = record_attributes[-1]
    weight = float(weight_str) if is_float_regex(weight_str) else 0.0
    
    if weight < 50:
        return None

    return dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=entity_description,
        edge_role=edge_role,
        weight=weight,
        hyper_relation=now_hyper_relation,
        source_id=chunk_key,
    )


async def _handle_single_hyperrelation_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 3 or record_attributes[0] != '"hyper-relation"':
        return None
    
    weight_str = record_attributes[-1]
    weight = float(weight_str) if is_float_regex(weight_str) else 0.0
    
    if weight < 7:
        return None

    knowledge_fragment = clean_str(record_attributes[1])
    
    return dict(
        hyper_relation="<hyperedge>"+knowledge_fragment,
        weight=weight,
        source_id=chunk_key,
    )
    

async def _merge_hyperedges_then_upsert(
    hyperedge_name: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
    paper_name: str = None, 
):
    already_weights = []
    already_source_ids = []

    already_hyperedge = await knowledge_graph_inst.get_node(hyperedge_name)
    if already_hyperedge is not None:
        already_weights.append(already_hyperedge["weight"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_hyperedge["source_id"], [GRAPH_FIELD_SEP])
        )

    weight = sum([dp["weight"] for dp in nodes_data] + already_weights)
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in nodes_data] + already_source_ids)
    )
    node_data = dict(
        role="hyperedge",
        weight=weight,
        source_id=source_id,
    )

    await knowledge_graph_inst.upsert_node(hyperedge_name, node_data=node_data)

    if paper_name:
        await knowledge_graph_inst.upsert_edge(
            paper_name,  
            hyperedge_name, 
            edge_data=dict(
                role="EVIDENCE", 
                description="BELONG_TO",  
                weight=1.0,
                source_id=source_id  
            ),
        )

    node_data["hyperedge_name"] = hyperedge_name
    return node_data


async def _get_hybrid_bm25_vector_results(query_text, all_chunks, vector_results, top_k=10):
    tokenized_corpus = [list(jieba.cut(doc)) for doc in all_chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = list(jieba.cut(query_text))
    
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_ranked_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:50]
    bm25_results = [all_chunks[i] for i in bm25_ranked_indices]
    
    rrf_scores = {}
    for rank, doc in enumerate(bm25_results):
        rrf_scores[doc] = rrf_scores.get(doc, 0.0) + 1.0 / (60 + rank + 1)
        
    for rank, doc in enumerate(vector_results): 
        rrf_scores[doc] = rrf_scores.get(doc, 0.0) + 1.0 / (60 + rank + 1)
        
    final_sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in final_sorted_docs[:top_k]]


async def _merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    already_entity_types = []
    already_source_ids = []
    already_description = []

    already_node = await knowledge_graph_inst.get_node(entity_name)
    if already_node is not None:
        already_entity_types.append(already_node["entity_type"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_node["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_node["description"])

    entity_type = sorted(
        Counter([dp["entity_type"] for dp in nodes_data] + already_entity_types).items(),
        key=lambda x: x[1],
        reverse=True,
    )[0][0]
    
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in nodes_data] + already_description))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in nodes_data] + already_source_ids)
    )
    description = await _handle_entity_relation_summary(
        entity_name, description, global_config
    )
    node_data = dict(
        role="entity",
        entity_type=entity_type,
        description=description,
        source_id=source_id,
    )
    await knowledge_graph_inst.upsert_node(entity_name, node_data=node_data)
    node_data["entity_name"] = entity_name
    return node_data


async def _merge_edges_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    edge_data = []
    for node in nodes_data:
        source_id = node["source_id"]
        hyper_relation = node["hyper_relation"]
        weight = node["weight"]
        role = node.get("edge_role", "CONTEXT")
        
        already_weights = []
        already_source_ids = []
        
        if await knowledge_graph_inst.has_edge(hyper_relation, entity_name):
            already_edge = await knowledge_graph_inst.get_edge(hyper_relation, entity_name)
            already_weights.append(already_edge["weight"])
            already_source_ids.extend(
                split_string_by_multi_markers(already_edge["source_id"], [GRAPH_FIELD_SEP])
            )
        
        weight = sum([weight] + already_weights)
        source_id = GRAPH_FIELD_SEP.join(set([source_id] + already_source_ids))

        await knowledge_graph_inst.upsert_edge(
            hyper_relation,
            entity_name,
            edge_data=dict(
                weight=weight,
                source_id=source_id,
                role=role,
                description="RELATES_TO",
            ),
        )

        edge_data.append(dict(
            src_id=hyper_relation,
            tgt_id=entity_name,
            weight=weight,
            role=role 
        ))

    return edge_data


def _compute_hyperedge_semantic_overlap(embs_a: list, embs_b: list, threshold: float = 0.85) -> float:
    """Mean of max cosine similarity from each element in the *smaller* set to the *larger* set.

    Returns a score in [0, 1] indicating how much the two hyperedge embedding sets overlap.
    """
    if not embs_a or not embs_b:
        return 0.0
    if len(embs_a) > len(embs_b):
        embs_a, embs_b = embs_b, embs_a

    max_sims = []
    for emb_a in embs_a:
        best = 0.0
        norm_a = float(np.linalg.norm(emb_a))
        if norm_a == 0:
            continue
        for emb_b in embs_b:
            norm_b = float(np.linalg.norm(emb_b))
            if norm_b == 0:
                continue
            cos = float(np.dot(emb_a, emb_b) / (norm_a * norm_b))
            best = max(best, cos)
        max_sims.append(best)

    return float(np.mean(max_sims)) if max_sims else 0.0


def _transitive_closure(merge_map: dict[str, str]) -> dict[str, str]:
    """Compute the transitive closure of a merge mapping and return {variant: root} for every key."""
    if not merge_map:
        return {}

    parent = dict(merge_map)

    def find_root(x: str) -> str:
        path = []
        while x in parent:
            path.append(x)
            x = parent[x]
        root = x
        for node in path:
            parent[node] = root
        return root

    result: dict[str, str] = {}
    all_names = set(merge_map.keys()) | set(merge_map.values())
    for name in all_names:
        root = find_root(name)
        if root != name:
            result[name] = root
    return result


async def _resolve_entity_cooccurrence(
    maybe_nodes: dict[str, list[dict]],
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    global_config: dict,
) -> dict[str, str]:
    """Co-occurrence constrained entity resolution.

    Two entities are merged ONLY when BOTH conditions hold:
    1. Name embeddings are similar (cos >= *cos_threshold*), AND
    2. They share sufficient hyperedge context (structural evidence).

    This prevents merging clinically distinct but semantically similar entities
    (e.g. pMMR vs dMMR) while unifying variant spellings / abbreviations
    (e.g. VBT vs Vaginal Brachytherapy).

    Returns a mapping ``{variant_name: canonical_name}``.
    """
    if not maybe_nodes or len(maybe_nodes) < 2:
        return {}

    embedding_func = entity_vdb.embedding_func
    if embedding_func is None:
        return {}

    cos_threshold = float(global_config.get("entity_resolution_cos_threshold", 0.85))
    jaccard_threshold = float(global_config.get("entity_resolution_jaccard_threshold", 0.25))
    semantic_overlap_threshold = float(global_config.get("entity_resolution_semantic_overlap", 0.80))

    # ------------------------------------------------------------------
    # Build entity → hyperedge-name set from batch data
    # ------------------------------------------------------------------
    new_entity_hyperedges: dict[str, set] = {}
    for entity_name, data_list in maybe_nodes.items():
        hes: set = set()
        for d in data_list:
            hr = d.get("hyper_relation", "")
            if hr:
                hes.add(hr)
        new_entity_hyperedges[entity_name] = hes

    new_names = list(new_entity_hyperedges.keys())

    # Batch-embed all new entity names
    new_embeddings_list = await embedding_func(new_names)
    new_embeddings: dict[str, np.ndarray] = {
        name: emb for name, emb in zip(new_names, new_embeddings_list)
    }

    def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0

    merge_map: dict[str, str] = {}  # variant → canonical

    # ==================================================================
    # Tier 1 — Within-batch (exact hyperedge-name match)
    #
    # When two entities are extracted from the *same* chunk they often
    # share literal hyperedge names (e.g. "VBT" and "Vaginal
    # Brachytherapy" both linked to the same knowledge-fragment).
    # ==================================================================
    for i in range(len(new_names)):
        name_i = new_names[i]
        if name_i in merge_map:
            continue
        hes_i = new_entity_hyperedges[name_i]
        if not hes_i:
            continue

        for j in range(i + 1, len(new_names)):
            name_j = new_names[j]
            if name_j in merge_map:
                continue
            hes_j = new_entity_hyperedges[name_j]
            if not hes_j:
                continue

            if _cos_sim(new_embeddings[name_i], new_embeddings[name_j]) < cos_threshold:
                continue

            shared = len(hes_i & hes_j)
            if shared == 0:
                continue

            jaccard = shared / len(hes_i | hes_j)
            if jaccard >= jaccard_threshold:
                # Prefer the longer (more explicit) name as canonical
                canonical = name_i if len(name_i) >= len(name_j) else name_j
                variant = name_j if canonical == name_i else name_i
                merge_map[variant] = canonical
                new_entity_hyperedges[canonical] |= hes_j

    # ==================================================================
    # Tier 2 — Cross-batch (semantic hyperedge-content overlap)
    #
    # An entity extracted now may correspond to one already in the graph
    # under a different surface form.  We compare the two *hyperedge
    # neighbourhoods* via their embeddings — not just names — to decide
    # whether they are the same clinical concept.
    # ==================================================================
    already_resolved = set(merge_map.keys()) | set(merge_map.values())
    unresolved = [n for n in new_names if n not in already_resolved]

    for name in unresolved:
        hes_new = new_entity_hyperedges.get(name, set())
        if not hes_new:
            continue

        try:
            similar_existing = await entity_vdb.search(
                query_vector=new_embeddings[name], top_k=5, score_threshold=cos_threshold
            )
        except Exception:
            continue

        for candidate in similar_existing:
            existing_name = candidate.get("entity_name", "")
            if not existing_name or existing_name == name:
                continue
            if existing_name in merge_map:
                continue

            # --- fetch existing node's hyperedge neighbours from graph ---
            try:
                existing_edges = await knowledge_graph_inst.get_node_edges_with_roles(existing_name)
            except Exception:
                continue

            if not existing_edges:
                continue

            existing_he_names: set = set()
            for edge in existing_edges:
                neighbour = edge[1] if len(edge) > 1 else ""
                if neighbour:
                    existing_he_names.add(neighbour)

            # -- fast path: literal hyperedge-name match --
            if hes_new & existing_he_names:
                merge_map[name] = existing_name
                break

            # -- slow path: semantic overlap between hyperedge-sets --
            he_new_list = list(hes_new)
            he_existing_list = list(existing_he_names)

            try:
                he_new_embs = await embedding_func(he_new_list)
                he_existing_embs = await embedding_func(he_existing_list)
            except Exception:
                continue

            overlap = _compute_hyperedge_semantic_overlap(
                he_new_embs, he_existing_embs, threshold=0.85
            )
            if overlap >= semantic_overlap_threshold:
                merge_map[name] = existing_name
                break

    return _transitive_closure(merge_map)


@traced("extract_entities")
async def extract_entities(
    chunks: dict[str, TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    hyperedge_vdb: BaseVectorStorage,
    global_config: dict,
    paper_name: str = None, 
) -> Union[BaseGraphStorage, None]:

    use_llm_func: callable = global_config["llm_model_func"]
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]
    ordered_chunks = list(chunks.items())
    language = global_config["addon_params"].get("language", PROMPTS["DEFAULT_LANGUAGE"])
    entity_types = global_config["addon_params"].get("entity_types", PROMPTS["DEFAULT_ENTITY_TYPES"])
    
    example_number = global_config["addon_params"].get("example_number", None)
    if example_number and example_number < len(PROMPTS["entity_extraction_examples"]):
        examples = "\n".join(PROMPTS["entity_extraction_examples"][: int(example_number)])
    else:
        examples = "\n".join(PROMPTS["entity_extraction_examples"])

    example_context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(entity_types),
        language=language,
    )
    examples = examples.format(**example_context_base)

    entity_extract_prompt = PROMPTS["entity_extraction"]
    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        examples=examples,
        language=language,
    )

    continue_prompt = PROMPTS["entiti_continue_extraction"]
    if_loop_prompt = PROMPTS["entiti_if_loop_extraction"]

    already_processed = 0
    already_entities = 0
    already_relations = 0

    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        nonlocal already_processed, already_entities, already_relations
        chunk_key, chunk_dp = chunk_key_dp
        content = chunk_dp["content"]
        hint_prompt = entity_extract_prompt.format(**context_base, input_text="{input_text}").format(**context_base, input_text=content)

        final_result = await use_llm_func(hint_prompt)
        history = pack_user_ass_to_openai_messages(hint_prompt, final_result)
        for now_glean_index in range(entity_extract_max_gleaning):
            glean_result = await use_llm_func(continue_prompt, history_messages=history)
            history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)
            final_result += glean_result
            if now_glean_index == entity_extract_max_gleaning - 1:
                break
            if_loop_result: str = await use_llm_func(if_loop_prompt, history_messages=history)
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            if if_loop_result != "yes":
                break

        records = split_string_by_multi_markers(
            final_result,
            [context_base["record_delimiter"], context_base["completion_delimiter"]],
        )

        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)
        now_hyper_relation=""
        for record in records:
            record = re.search(r"\((.*)\)", record)
            if record is None: continue
            record = record.group(1)
            record_attributes = split_string_by_multi_markers(record, [context_base["tuple_delimiter"]])
            
            if len(record_attributes) > 0 and record_attributes[0] == '"hyper-relation"':
                if_relation = await _handle_single_hyperrelation_extraction(record_attributes, chunk_key)
                if if_relation is not None:
                    maybe_edges[if_relation["hyper_relation"]].append(if_relation)
                    now_hyper_relation = if_relation["hyper_relation"]
                else:
                    now_hyper_relation = ""
                continue
            
            if_entities = await _handle_single_entity_extraction(record_attributes, chunk_key, now_hyper_relation)
            if if_entities is not None:
                maybe_nodes[if_entities["entity_name"]].append(if_entities)
                continue
            
        already_processed += 1
        already_entities += len(maybe_nodes)
        already_relations += len(maybe_edges)
        now_ticks = PROMPTS["process_tickers"][already_processed % len(PROMPTS["process_tickers"])]
        print(f"{now_ticks} Processed {already_processed} chunks, {already_entities} entities(duplicated), {already_relations} relations(duplicated)\r", end="", flush=True)
        return dict(maybe_nodes), dict(maybe_edges)

    results = []
    for result in tqdm_async(
        asyncio.as_completed([_process_single_content(c) for c in ordered_chunks]),
        total=len(ordered_chunks), desc="Extracting entities from chunks", unit="chunk"
    ):
        results.append(await result)

    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)
    for m_nodes, m_edges in results:
        for k, v in m_nodes.items(): maybe_nodes[k].extend(v)
        for k, v in m_edges.items(): maybe_edges[k].extend(v)

    # ── Co-occurrence constrained entity resolution ──
    if global_config.get("entity_resolution_enabled", True):
        logger.info("Running co-occurrence constrained entity resolution...")
        merge_map = await _resolve_entity_cooccurrence(
            maybe_nodes, knowledge_graph_inst, entity_vdb, global_config
        )
        if merge_map:
            for variant, canonical in merge_map.items():
                if variant not in maybe_nodes:
                    continue
                variant_data = maybe_nodes.pop(variant)
                for entry in variant_data:
                    entry["entity_name"] = canonical
                if canonical in maybe_nodes:
                    maybe_nodes[canonical].extend(variant_data)
                else:
                    maybe_nodes[canonical] = variant_data
                logger.info(
                    "🔗 [Entity Resolution] Merged '%s' -> '%s'", variant, canonical
                )

    write_concurrency_limit = asyncio.Semaphore(30)
    async def _sem_task(coro):
        async with write_concurrency_limit:
            return await coro

    logger.info("Inserting hyperedges into storage...")
    all_hyperedges_data = []
    tasks = [_sem_task(_merge_hyperedges_then_upsert(k, v, knowledge_graph_inst, global_config, paper_name=paper_name)) for k, v in maybe_edges.items()]
    for result in tqdm_async(asyncio.as_completed(tasks), total=len(maybe_edges), desc="Inserting hyperedges", unit="entity"):
        all_hyperedges_data.append(await result)
            
    logger.info("Inserting entities into storage...")
    all_entities_data = []
    tasks = [_sem_task(_merge_nodes_then_upsert(k, v, knowledge_graph_inst, global_config)) for k, v in maybe_nodes.items()]
    for result in tqdm_async(asyncio.as_completed(tasks), total=len(maybe_nodes), desc="Inserting entities", unit="entity"):
        all_entities_data.append(await result)

    logger.info("Inserting relationships into storage...")
    all_relationships_data = []
    tasks = [_sem_task(_merge_edges_then_upsert(k, v, knowledge_graph_inst, global_config)) for k, v in maybe_nodes.items()]
    for result in tqdm_async(asyncio.as_completed(tasks), total=len(maybe_nodes), desc="Inserting relationships", unit="relationship"):
        all_relationships_data.append(await result)

    if not len(all_hyperedges_data) and not len(all_entities_data) and not len(all_relationships_data):
        logger.warning("Didn't extract any hyperedges and entities, maybe your LLM is not working")
        return None

    if hyperedge_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["hyperedge_name"], prefix="rel-"): {
                "content": dp["hyperedge_name"],
                "hyperedge_name": dp["hyperedge_name"],
            } for dp in all_hyperedges_data
        }
        await hyperedge_vdb.upsert(data_for_vdb)

    if entity_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                "content": dp["entity_name"] + dp["description"],
                "entity_name": dp["entity_name"],
            } for dp in all_entities_data
        }
        await entity_vdb.upsert(data_for_vdb)

    return knowledge_graph_inst


@traced("kg_query")
async def kg_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: list,
    hyperedges_vdb: list,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
    hashing_kv: BaseKVStorage = None,
) -> str:
    hl_keywords, ll_keywords = query, query
    keywords = [ll_keywords, hl_keywords]
    context = await _build_query_context(
        keywords, knowledge_graph_inst, entities_vdb, hyperedges_vdb,
        text_chunks_db, query_param, global_config
    )
    return context


async def _build_query_context(
    query: list,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    hyperedges_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict, 
):
    ll_kewwords = query[0]
    hl_keywrds = query[1]
    moe_g_weight = getattr(query_param, "moe_weight", 0.5)
    ablation_mode = global_config.get("addon_params", {}).get("ablation_mode", "full")

    # 1. 向量单跳流
    with trace_span("vector_retrieval"):
        vector_list = await _get_edge_data(
            hl_keywrds, knowledge_graph_inst, hyperedges_vdb, text_chunks_db, query_param, global_config
        )

    # 2. 消融实验分支
    if ablation_mode == "bm25_vector_only":
        if hasattr(text_chunks_db, "filter"):
            all_data_dict = await text_chunks_db.filter(lambda _: True)
        else:
            all_data_dict = text_chunks_db._data 
            
        all_chunks = [v["content"] for v in all_data_dict.values() if v and "content" in v]
        bm25_results = await _get_hybrid_bm25_vector_results(ll_kewwords, all_chunks, vector_list, top_k=query_param.top_k)
        return [{"<knowledge>": k, "<coherence>": 0.8} for k in bm25_results]

    elif ablation_mode == "vector_only":
        graph_dict = {}  
    else:
        with trace_span("entity_graph_walk"):
            graph_dict = await _get_node_data(
                ll_kewwords, knowledge_graph_inst, entities_vdb, text_chunks_db, query_param, global_config
            )

    # 3. 终极算法：纯粹的 MoE-RRF (已移除硬置顶逻辑)
    final_candidates = {}
    graph_list = [item[0] for item in sorted(graph_dict.items(), key=lambda x: x[1], reverse=True)] if graph_dict else []

    RRF_K = 60.0
    vector_weight = 1.0 - moe_g_weight
    graph_weight = moe_g_weight

    # A. 向量列表 RRF
    for rank, content in enumerate(vector_list):
        if content not in final_candidates:
            final_candidates[content] = {"score": 0.0, "sources": []}
        final_candidates[content]["score"] += vector_weight * (1.0 / (RRF_K + rank + 1))
        final_candidates[content]["sources"].append("vector")

    # B. 图谱列表 RRF 
    for rank, content in enumerate(graph_list):
        if content not in final_candidates:
            final_candidates[content] = {"score": 0.0, "sources": []}
        final_candidates[content]["score"] += graph_weight * (1.0 / (RRF_K + rank + 1))
        final_candidates[content]["sources"].append("graph")
        
    # C. 双路共识奖励 (1.2倍)
    for content, data in final_candidates.items():
        if len(data["sources"]) == 2:
            data["score"] *= 1.2

    sorted_candidates = sorted(final_candidates.items(), key=lambda x: x[1]["score"], reverse=True)[:query_param.top_k]
    knowledge = [{"<knowledge>": k, "<coherence>": round(v["score"], 4)} for k, v in sorted_candidates]
    return knowledge


async def _get_node_data(
    query, 
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict, 
):  
    results = entities_vdb
    if not len(results):
        return {} 

    node_keys = [r["entity_name"] if isinstance(r, dict) else r for r in results]
    node_datas = await _gather_reads([knowledge_graph_inst.get_node(k) for k in node_keys])
    
    if not all([n is not None for n in node_datas]):
        logger.warning("Some nodes are missing, maybe the storage is damaged")

    node_degrees = await _gather_reads([knowledge_graph_inst.node_degree(k) for k in node_keys])
    
    node_datas = [
        {**n, "entity_name": k, "rank": d}
        for k, n, d in zip(node_keys, node_datas, node_degrees) if n is not None
    ]  
    
    use_relations = await _find_most_related_edges_from_entities(
        node_datas, query_param, knowledge_graph_inst, query, global_config
    )

    print("\n" + "="*20 + " 🚀 [超图多维特征同构召回] " + "="*20)
    for idx, edge in enumerate(use_relations[:5]): 
        matched_ents = edge.get('matched_entities', [])
        score = edge.get('coverage_score', 0)
        desc_snippet = edge.get('description', '')[:60].replace('\n', ' ') 
        print(f"Top {idx+1} | 匹配得分: {score:.4f} | 命中实体: {matched_ents}")
        print(f"  └─ 方案内容: {desc_snippet}...")
    print("="*65 + "\n")

    knowledge_dict = {}
    for s in use_relations:
        desc = s["description"].replace("<hyperedge>", "")
        knowledge_dict[desc] = s.get("coverage_score", 0.0)

    return knowledge_dict


async def _find_most_related_edges_from_entities(
    node_datas: list[dict],
    query_param: QueryParam, 
    knowledge_graph_inst,    
    query_str: str,       
    global_config: dict   
):
    MAX_NEIGHBORS_PER_NODE = 50 

    # ================= 阶段一：前置查询向量 =================
    embedding_func = knowledge_graph_inst.embedding_func
    query_emb_np = None
    if embedding_func:
        query_emb_np = (await embedding_func([query_str]))[0]

    def calc_cosine_sim(v1, v2):
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        return float(np.dot(v1, v2) / norm) if norm > 0 else 0.0

    # ================= 阶段二：柔性游走 =================
    all_related_edges_raw = []
    hard_edges_results = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges_with_roles(dp["entity_name"]) for dp in node_datas]
    )
    
    for i, (dp, this_edges) in enumerate(zip(node_datas, hard_edges_results)):
        entity_name = dp["entity_name"]
        
        if this_edges and len(this_edges) > 0:
            all_related_edges_raw.append(this_edges)
            continue
            
        print(f"\n🔗 [柔性游走]: 节点 【{entity_name}】 物理断链，启动空间向量吸附...")
        soft_edges = []
        
        if embedding_func and hasattr(knowledge_graph_inst, "entities_vdb"):
            try:
                ent_emb = (await embedding_func([entity_name]))[0]
                nearest_entities = await knowledge_graph_inst.entities_vdb.search(
                    query_vector=ent_emb, top_k=3, score_threshold=0.85 
                )
                
                for neighbor in nearest_entities:
                    neighbor_name = neighbor["entity_name"]
                    sim_score = neighbor["score"]
                    if neighbor_name != entity_name:
                        soft_edges.append((entity_name, neighbor_name, "CONTEXT", sim_score))
                        print(f"   -> 🧲 吸附到语义相邻节点: 【{neighbor_name}】")
            except Exception as e:
                pass
                
        all_related_edges_raw.append(soft_edges)

    hyperedge_to_entities = defaultdict(list)
    entity_to_hyperedges = defaultdict(list) 
    
    for dp, this_edges in zip(node_datas, all_related_edges_raw):
        if not this_edges: continue
        entity_name = dp["entity_name"]
        
        # 🛡️ 截断保护 (保留核心方案，其余按相似度淘汰)
        if len(this_edges) > MAX_NEIGHBORS_PER_NODE:
            must_keep = []
            candidates = []
            
            for e in this_edges:
                role = e[2] if len(e) > 2 else "UNKNOWN"
                if role in ["RECOMMENDATION"]:
                    must_keep.append(e)
                else:
                    candidates.append(e)
            
            quota = MAX_NEIGHBORS_PER_NODE - len(must_keep)
            if quota > 0 and len(candidates) > quota and query_emb_np is not None:
                cand_texts = [e[1] for e in candidates]
                cand_embs = await embedding_func(cand_texts)
                cand_with_sim = [(cand, calc_cosine_sim(query_emb_np, emb)) for cand, emb in zip(candidates, cand_embs)]
                cand_with_sim.sort(key=lambda x: (x[1], x[0][1]), reverse=True)
                candidates = [x[0] for x in cand_with_sim[:quota]]
            elif quota <= 0:
                candidates = []
            elif query_emb_np is None and quota > 0:
                candidates = sorted(candidates, key=lambda x: -(x[-1] if isinstance(x[-1], float) else 1.0))[:quota]

            this_edges = must_keep + candidates

        for e in this_edges:
            he_name = e[1]
            role = e[2] if len(e) > 2 else "UNKNOWN"
            hyperedge_to_entities[he_name].append((entity_name, role))
            entity_to_hyperedges[entity_name].append(he_name)

    unique_hyperedges = sorted(list(hyperedge_to_entities.keys()))

    # ================= 阶段三：获取二阶邻居聚合 =================
    hyperedge_neighbors_raw = await _gather_reads(
        [knowledge_graph_inst.get_node_edges(he) for he in unique_hyperedges]
    )

    parent_aggregation = defaultdict(lambda: {"contained_hyperedges": set(), "matched_items": set()})
    
    for he, neighbors in zip(unique_hyperedges, hyperedge_neighbors_raw):
        if not neighbors: continue
            
        if len(neighbors) > MAX_NEIGHBORS_PER_NODE:
            neighbors = sorted(neighbors, key=lambda x: -(x[-1] if isinstance(x[-1], float) else 1.0))[:MAX_NEIGHBORS_PER_NODE]
            
        for edge in neighbors:
            neighbor_name = edge[1]
            if neighbor_name.startswith("paper::"): 
                parent_aggregation[neighbor_name]["contained_hyperedges"].add(he)
                parent_aggregation[neighbor_name]["matched_items"].update(hyperedge_to_entities[he])

    # ================= 阶段四：组装 HGNN 张量 =================
    ROLE_VOCAB = {"EVIDENCE": 0, "CONTEXT": 1, "CONDITION": 2, "RECOMMENDATION": 3, "CONTRAINDICATION": 4, "PAD": 5}
    MAX_ENTITIES = 40 
    
    hyperedge_tensors, hyperedge_roles, hyperedge_masks, valid_hyperedges = [], [], [], []
    hyperedge_role_profiles = defaultdict(set) 
    
    for he in unique_hyperedges:
        hit_items = hyperedge_to_entities[he]
        embs, roles, masks = [], [], []
        
        for ent, role in hit_items:
            hyperedge_role_profiles[he].add(role)
            ent_key = ent.upper()
            if ent_key in GLOBAL_ENTITY_CACHE:
                embs.append(GLOBAL_ENTITY_CACHE[ent_key])
                roles.append(ROLE_VOCAB.get(role.upper(), 1)) 
                masks.append(1.0)

        if len(embs) > 0:
            embs = embs[:MAX_ENTITIES]
            roles = roles[:MAX_ENTITIES]
            masks = masks[:MAX_ENTITIES]
            
            pad_len = MAX_ENTITIES - len(embs)
            if pad_len > 0:
                dim = embs[0].size(0)
                embs.extend([torch.zeros(dim)] * pad_len)
                roles.extend([ROLE_VOCAB["PAD"]] * pad_len)
                masks.extend([0.0] * pad_len)
                
            hyperedge_tensors.append(torch.stack(embs))
            hyperedge_roles.append(roles)
            hyperedge_masks.append(masks)
            valid_hyperedges.append(he)

    # ================= 阶段五：调用 HGNN 推断 =================
    hyperedge_scores = {}
    
    if valid_hyperedges and query_emb_np is not None:
        he_embs_batch = torch.stack(hyperedge_tensors)
        he_roles_batch = torch.tensor(hyperedge_roles, dtype=torch.long)
        he_masks_batch = torch.tensor(hyperedge_masks, dtype=torch.bool)
        query_emb_tensor = torch.tensor(query_emb_np, dtype=torch.float32)
        
        hg_scores = await asyncio.to_thread(
            compute_hyperedge_scores_sync, 
            query_emb_tensor, 
            (he_embs_batch, he_roles_batch, he_masks_batch)
        )
        
        for he, score in zip(valid_hyperedges, hg_scores):
            hyperedge_scores[he] = float(score)
    else:
        print("⚠️ 警告: 缺少 embedding_func 或没有有效超边，打分降级为 0")

    for he in unique_hyperedges:
        if he not in hyperedge_scores:
            hyperedge_scores[he] = 0.0

    # ================= 阶段六：分层聚合排序 =================
    parent_scores = []
    for parent, data in parent_aggregation.items():
        contained_hes = sorted(list(data["contained_hyperedges"]))
        matched_items = sorted(list(data["matched_items"]), key=lambda x: x[0])
        valid_he_scores = [hyperedge_scores[he] for he in contained_hes if hyperedge_scores[he] > 0]

        if valid_he_scores:
            coverage_score = sum(valid_he_scores)
            
            has_recommendation = any("RECOMMENDATION" in hyperedge_role_profiles.get(he, set()) for he in contained_hes)
            has_evidence = any("EVIDENCE" in hyperedge_role_profiles.get(he, set()) for he in contained_hes)
            tier = 3 if has_recommendation else (2 if has_evidence else 1)

            parent_scores.append({
                "parent_name": parent,
                "coverage_score": round(coverage_score, 4),
                "tier": tier,
                "contained_hyperedges": list(contained_hes),
                "matched_items": matched_items
            })

    parent_scores = sorted(
        parent_scores,
        key=lambda x: (x["tier"], x["coverage_score"], len(x["contained_hyperedges"])),
        reverse=True
    )
    top_parents = parent_scores[:5] 

    # ================= 阶段七：控制台打印与组装 =================
    print("\n" + "="*15 + " 🧮 [神经推演明细] " + "="*15)
    for i, p in enumerate(top_parents):
        print(f"[{i+1}] {p['parent_name']} | 逻辑层级: Tier {p['tier']} | ∑推荐度: {p['coverage_score']:.4f}")
        for he in p["contained_hyperedges"]:
            score = hyperedge_scores[he]
            hit_items = hyperedge_to_entities[he]
            detail_strs = [ent for ent, role in hit_items]
            he_snippet = he[:25].replace('\n', '') + "..." 
            trend_str = f"🚀 {score:.2%}" if score > 0.7 else f"🔻 {score:.2%}" if score < 0.3 else f"⚖️ {score:.2%}"
            print(f"    ├─ [{list(hyperedge_role_profiles[he])[0]}] 推荐度: {trend_str} | {he_snippet}")
            print(f"    │   └─ 命中特征: {' + '.join(detail_strs)}")
    print("="*50 + "\n")

    all_edges_data = []
    for p in top_parents:
        aggregated_desc = f"【权威循证溯源：{p['parent_name']}】(逻辑层级: Tier {p['tier']}, 推荐度: {p['coverage_score']:.4f})\n"
        
        intra_group_edges = []
        for he in p["contained_hyperedges"]:
            if hyperedge_scores[he] > 0:
                hit_items = hyperedge_to_entities[he] 
                intra_group_edges.append((he, hyperedge_scores[he], hit_items))
            
        intra_group_edges.sort(key=lambda x: x[1], reverse=True)
        kept_edges = intra_group_edges[:4] 
        
        for idx, (he, sat_score, hit_items) in enumerate(kept_edges):
            trigger_reason = ", ".join([f"{ent}" for ent, role in hit_items])
            he_roles = "/".join(list(hyperedge_role_profiles[he]))
            aggregated_desc += (
                f"  > 候选路径 {idx+1} [{he_roles}] (神经推演推荐度: {sat_score:.2%})\n"
                f"    ├─ 方案内容: {he}\n"
                f"    └─ 触发该方案的局部特征: 【{trigger_reason}】\n"
            )

        all_edges_data.append({
            "description": aggregated_desc,
            "coverage_score": p["coverage_score"], 
            "rank": len(p["contained_hyperedges"]),
            "matched_entities": [item[0] for item in p["matched_items"]], 
            "weight": 1.0
        })

    return all_edges_data

async def _get_edge_data(
    query_str, 
    knowledge_graph_inst: BaseGraphStorage,
    hyperedges_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict, 
):  
    results = hyperedges_vdb
    if not len(results): return []

    edge_keys = [r["hyperedge_name"] if isinstance(r, dict) else r for r in results]
    edge_datas = await _gather_reads([knowledge_graph_inst.get_node(k) for k in edge_keys])
    
    edge_datas = [{"hyperedge": k, "rank": v["weight"], **v} 
                  for k, v in zip(edge_keys, edge_datas) if v is not None]
    
    docs_to_rerank = [s["hyperedge"].replace("<hyperedge>","") for s in edge_datas]

    reranker_func = global_config.get("reranker_func", None)
    if reranker_func and docs_to_rerank:
        knowledge_list = await reranker_func(query_str, docs_to_rerank) 
    else:
        edge_datas = sorted(edge_datas, key=lambda x: (x["rank"], x["weight"]), reverse=True)
        knowledge_list = [s["hyperedge"].replace("<hyperedge>","") for s in edge_datas]
        
    return knowledge_list


async def _find_most_related_entities_from_relationships(
    edge_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):
    node_datas = await _gather_reads(
        [knowledge_graph_inst.get_node_edges(edge["hyperedge"]) for edge in edge_datas]
    )
    
    entity_names = []
    seen = set()

    for node_data in node_datas:
        for e in node_data:
            if e[1] not in seen:
                entity_names.append(e[1])
                seen.add(e[1])

    node_datas = await _gather_reads([knowledge_graph_inst.get_node(entity_name) for entity_name in entity_names])
    node_degrees = await _gather_reads([knowledge_graph_inst.node_degree(entity_name) for entity_name in entity_names])
    
    node_datas = [
        {**n, "entity_name": k, "rank": d}
        for k, n, d in zip(entity_names, node_datas, node_degrees)
    ]

    node_datas = truncate_list_by_token_size(
        node_datas,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_local_context,
    )

    return node_datas


async def _find_related_text_unit_from_relationships(
    edge_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
):
    text_units = [split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP]) for dp in edge_datas]
    all_text_units_lookup = {}

    for index, unit_list in enumerate(text_units):
        for c_id in unit_list:
            if c_id not in all_text_units_lookup:
                chunk_data = await text_chunks_db.get_by_id(c_id)
                if chunk_data is not None and "content" in chunk_data:
                    all_text_units_lookup[c_id] = {
                        "data": chunk_data,
                        "order": index,
                    }

    if not all_text_units_lookup:
        logger.warning("No valid text chunks found")
        return []

    all_text_units = [{"id": k, **v} for k, v in all_text_units_lookup.items()]
    all_text_units = sorted(all_text_units, key=lambda x: x["order"])

    valid_text_units = [t for t in all_text_units if t["data"] is not None and "content" in t["data"]]

    if not valid_text_units:
        return []

    truncated_text_units = truncate_list_by_token_size(
        valid_text_units,
        key=lambda x: x["data"]["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )

    all_text_units: list[TextChunkSchema] = [t["data"] for t in truncated_text_units]

    return all_text_units


def combine_contexts(entities, relationships, sources):
    hl_entities, ll_entities = entities[0], entities[1]
    hl_relationships, ll_relationships = relationships[0], relationships[1]
    combined_entities = process_combine_contexts(hl_entities, ll_entities)
    combined_relationships = process_combine_contexts(hl_relationships, ll_relationships)
    return combined_entities, combined_relationships, ""