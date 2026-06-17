# pathorag_core/pathorag.py
import asyncio
import os
from tqdm.asyncio import tqdm as tqdm_async
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial
from typing import Type, cast

from .llm import (
    gpt_4o_mini_complete,
    openai_embedding,
)
from .operate import (
    chunking_by_token_size,
    chunking_by_markdown_headers,
    extract_entities,
    kg_query
)

from .instrumentation import (
    init_instrumentation,
    traced,
    trace_span,
    inc_counter,
    observe_histogram,
    inc_gauge,
    dec_gauge,
)
from .utils import (
    EmbeddingFunc,
    UnlimitedSemaphore,
    OverloadError,
    compute_mdhash_id,
    limit_async_func_call,
    convert_response_to_json,
    logger,
    set_logger,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    StorageNameSpace,
    QueryParam,
)

from .storage import (
    JsonKVStorage,
    NanoVectorDBStorage,
    NetworkXStorage,
)

def lazy_external_import(module_name: str, class_name: str):
    """Lazily import a class from an external module based on the package of the caller."""
    import inspect
    caller_frame = inspect.currentframe().f_back
    module = inspect.getmodule(caller_frame)
    package = module.__package__ if module else None

    def import_class(*args, **kwargs):
        import importlib
        module = importlib.import_module(module_name, package=package)
        cls = getattr(module, class_name)
        return cls(*args, **kwargs)

    return import_class


Neo4JStorage = lazy_external_import(".kg.neo4j_impl", "Neo4JStorage")
OracleKVStorage = lazy_external_import(".kg.oracle_impl", "OracleKVStorage")
OracleGraphStorage = lazy_external_import(".kg.oracle_impl", "OracleGraphStorage")
OracleVectorDBStorage = lazy_external_import(".kg.oracle_impl", "OracleVectorDBStorage")
MilvusVectorDBStorge = lazy_external_import(".kg.milvus_impl", "MilvusVectorDBStorge")
MongoKVStorage = lazy_external_import(".kg.mongo_impl", "MongoKVStorage")
ChromaVectorDBStorage = lazy_external_import(".kg.chroma_impl", "ChromaVectorDBStorage")
TiDBKVStorage = lazy_external_import(".kg.tidb_impl", "TiDBKVStorage")
TiDBVectorDBStorage = lazy_external_import(".kg.tidb_impl", "TiDBVectorDBStorage")


@dataclass
class PathoRAG:
    working_dir: str = field(
        default_factory=lambda: f"pathorag_core_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )
    # Default not to use embedding cache
    embedding_cache_config: dict = field(
        default_factory=lambda: {
            "enabled": False,
            "similarity_threshold": 0.95,
            "use_llm_check": False,
        }
    )
    kv_storage: str = field(default="JsonKVStorage")
    vector_storage: str = field(default="NanoVectorDBStorage")
    graph_storage: str = field(default="NetworkXStorage")

    current_log_level = logger.level
    log_level: str = field(default=current_log_level)

    # text chunking
    chunking_mode: str = "semantic"  # "fixed" | "semantic"
    chunk_token_size: int = 1200
    chunk_overlap_token_size: int = 100
    tiktoken_model_name: str = "gpt-4o-mini"

    # semantic chunking (only used when chunking_mode="semantic")
    chunking_heading_level: int = 2  # max heading level to split on (1=#, 2=##, …)
    chunking_min_section_tokens: int = 0  # merge sections smaller than this (0=disabled)

    # entity extraction
    entity_extract_max_gleaning: int = 2
    entity_summary_to_max_tokens: int = 500

    # co-occurrence constrained entity resolution
    entity_resolution_enabled: bool = True
    entity_resolution_cos_threshold: float = 0.85
    entity_resolution_jaccard_threshold: float = 0.25
    entity_resolution_semantic_overlap: float = 0.80

    # node embedding
    node_embedding_algorithm: str = "node2vec"
    node2vec_params: dict = field(
        default_factory=lambda: {
            "dimensions": 1536,
            "num_walks": 10,
            "walk_length": 40,
            "window_size": 2,
            "iterations": 3,
            "random_seed": 3,
        }
    )

    # embedding_func: EmbeddingFunc = field(default_factory=lambda:hf_embedding)
    embedding_func: EmbeddingFunc = field(default_factory=lambda: openai_embedding)
    embedding_batch_num: int = 32
    embedding_func_max_async: int = 16

    # LLM
    llm_model_func: callable = gpt_4o_mini_complete  # hf_model_complete#
    llm_model_name: str = "meta-llama/Llama-3.2-1B-Instruct"  #'meta-llama/Llama-3.2-1B'#'google/gemma-2-2b-it'
    llm_model_max_token_size: int = 32768
    llm_model_max_async: int = 16
    llm_model_kwargs: dict = field(default_factory=dict)

    # Reranker (保留你最新添加的属性)
    reranker_func: callable = None

    # storage
    vector_db_storage_cls_kwargs: dict = field(default_factory=dict)

    enable_llm_cache: bool = True

    # extension
    addon_params: dict = field(default_factory=dict)
    convert_response_to_json_func: callable = convert_response_to_json

    # --- [工程鲁棒性新增] 系统级超时配置 ---
    db_timeout: float = 5.0
    llm_timeout: float = 45.0
    query_max_concurrent: int = 20  # 请求级并发上限
    instrumentation_enabled: bool = True
    instrumentation_otlp_endpoint: str = ""
    instrumentation_prometheus_port: int = 0

    def __post_init__(self):
        log_file = os.path.join("pathorag_core.log")
        set_logger(log_file)
        logger.setLevel(self.log_level)

        logger.info(f"Logger initialized for working directory: {self.working_dir}")

        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self).items()])
        logger.debug(f"PathoRAG init with param:\n  {_print_config}\n")

        self.key_string_value_json_storage_cls: Type[BaseKVStorage] = (
            self._get_storage_class()[self.kv_storage]
        )
        self.vector_db_storage_cls: Type[BaseVectorStorage] = self._get_storage_class()[
            self.vector_storage
        ]
        self.graph_storage_cls: Type[BaseGraphStorage] = self._get_storage_class()[
            self.graph_storage
        ]

        if not os.path.exists(self.working_dir):
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir)

        self.llm_response_cache = (
            self.key_string_value_json_storage_cls(
                namespace="llm_response_cache",
                global_config=asdict(self),
                embedding_func=None,
            )
            if self.enable_llm_cache
            else None
        )
        self.embedding_func.concurrent_limit = 0
        self.embedding_func._semaphore = UnlimitedSemaphore()
        self.embedding_func = limit_async_func_call(self.embedding_func_max_async)(
            self.embedding_func
        )

        self._query_semaphore = asyncio.Semaphore(self.query_max_concurrent)

        self.full_docs = self.key_string_value_json_storage_cls(
            namespace="full_docs",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )
        self.text_chunks = self.key_string_value_json_storage_cls(
            namespace="text_chunks",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )
        self.chunk_entity_relation_graph = self.graph_storage_cls(
            namespace="chunk_entity_relation",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )

        # ----------------------------------------------------------------------
        # [修复] 使用 vector_db_storage_cls 初始化向量库 (entities/hyperedges/chunks)
        # 原代码错误地使用了 key_string_value_json_storage_cls (KV存储)
        # ----------------------------------------------------------------------
        self.entities_vdb = self.vector_db_storage_cls(
            namespace="entities",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )
        self.hyperedges_vdb = self.vector_db_storage_cls(
            namespace="hyperedges",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )
        self.chunks_vdb = self.vector_db_storage_cls(
            namespace="chunks",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )
        # ----------------------------------------------------------------------

        self.llm_model_func = limit_async_func_call(self.llm_model_max_async)(
            partial(
                self.llm_model_func,
                hashing_kv=self.llm_response_cache,
                **self.llm_model_kwargs,
            )
        )

        if self.instrumentation_enabled:
            init_instrumentation(
                service_name="pathorag",
                otlp_endpoint=self.instrumentation_otlp_endpoint or None,
                console_export=True,
                prometheus_port=self.instrumentation_prometheus_port or None,
            )

    def _get_storage_class(self) -> Type[BaseGraphStorage]:
        return {
            # kv storage
            "JsonKVStorage": JsonKVStorage,
            "OracleKVStorage": OracleKVStorage,
            "MongoKVStorage": MongoKVStorage,
            "TiDBKVStorage": TiDBKVStorage,
            # vector storage
            "NanoVectorDBStorage": NanoVectorDBStorage,
            "OracleVectorDBStorage": OracleVectorDBStorage,
            "MilvusVectorDBStorge": MilvusVectorDBStorge,
            "ChromaVectorDBStorage": ChromaVectorDBStorage,
            "TiDBVectorDBStorage": TiDBVectorDBStorage,
            # graph storage
            "NetworkXStorage": NetworkXStorage,
            "Neo4JStorage": Neo4JStorage,
            "OracleGraphStorage": OracleGraphStorage,
            # "ArangoDBStorage": ArangoDBStorage
        }

    # ----------------------------------------------------------------------
    # [重构] 彻底剥离危险的强制全局事件循环 (防止 FastAPI/Uvicorn 死锁)
    # ----------------------------------------------------------------------
    def _run_sync(self, coro):
        """同步执行隔离包装器"""
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        if current_loop and current_loop.is_running():
            raise RuntimeError(
                "检测到处于异步框架中。请直接调用异步方法 (例如: await aquery())，"
                "切勿使用同步方法 (例如: query())，否则会导致线程阻塞或死锁。"
            )
        else:
            return asyncio.run(coro)

    def insert(self, string_or_strings, paper_name: str = None):
        return self._run_sync(self.ainsert(string_or_strings, paper_name))

    @traced("ainsert")
    async def ainsert(self, string_or_strings, paper_name: str = None):
        update_storage = False
        try:
            if isinstance(string_or_strings, str):
                string_or_strings = [string_or_strings]

            new_docs = {
                compute_mdhash_id(c.strip(), prefix="doc-"): {"content": c.strip()}
                for c in string_or_strings
            }
            _add_doc_keys = await self.full_docs.filter_keys(list(new_docs.keys()))
            new_docs = {k: v for k, v in new_docs.items() if k in _add_doc_keys}
            if not len(new_docs):
                logger.warning("All docs are already in the storage")
                return
            update_storage = True
            logger.info(f"[New Docs] inserting {len(new_docs)} docs")

            # ── Choose chunking strategy ──
            if self.chunking_mode == "semantic":
                _chunk_func = chunking_by_markdown_headers
                _chunk_kwargs = dict(
                    heading_level=self.chunking_heading_level,
                    min_section_tokens=self.chunking_min_section_tokens,
                )
            else:
                _chunk_func = chunking_by_token_size
                _chunk_kwargs = {}

            inserting_chunks = {}
            for doc_key, doc in tqdm_async(
                new_docs.items(), desc=f"Chunking documents ({self.chunking_mode})", unit="doc"
            ):
                chunks = {
                    compute_mdhash_id(dp["content"], prefix="chunk-"): {
                        **dp,
                        "full_doc_id": doc_key,
                    }
                    for dp in _chunk_func(
                        doc["content"],
                        overlap_token_size=self.chunk_overlap_token_size,
                        max_token_size=self.chunk_token_size,
                        tiktoken_model=self.tiktoken_model_name,
                        **_chunk_kwargs,
                    )
                }
                inserting_chunks.update(chunks)
            _add_chunk_keys = await self.text_chunks.filter_keys(
                list(inserting_chunks.keys())
            )
            inserting_chunks = {
                k: v for k, v in inserting_chunks.items() if k in _add_chunk_keys
            }
            if not len(inserting_chunks):
                logger.warning("All chunks are already in the storage")
                return
            logger.info(f"[New Chunks] inserting {len(inserting_chunks)} chunks")

            logger.info("[Entity Extraction]...")
            maybe_new_kg = await extract_entities(
                inserting_chunks,
                knowledge_graph_inst=self.chunk_entity_relation_graph,
                entity_vdb=self.entities_vdb,
                hyperedge_vdb=self.hyperedges_vdb,
                global_config=asdict(self),
                paper_name=paper_name, # 传递 paper_name 参数
            )
            if maybe_new_kg is None:
                logger.warning("No new hyperedges and entities found")
                return
            self.chunk_entity_relation_graph = maybe_new_kg

            await self.full_docs.upsert(new_docs)
            await self.text_chunks.upsert(inserting_chunks)
        finally:
            if update_storage:
                await self._insert_done()

    async def _insert_done(self):
        tasks = []
        for storage_inst in [
            self.full_docs,
            self.text_chunks,
            self.llm_response_cache,
            self.entities_vdb,
            self.hyperedges_vdb,
            self.chunks_vdb,
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        
        # [修改] return_exceptions=True 防异常穿透雪崩
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Task {idx} failed in _insert_done with error: {result}")

    def insert_custom_kg(self, custom_kg: dict):
        return self._run_sync(self.ainsert_custom_kg(custom_kg))

    async def ainsert_custom_kg(self, custom_kg: dict):
        update_storage = False
        try:
            # Insert chunks into vector storage
            all_chunks_data = {}
            chunk_to_source_map = {}
            for chunk_data in custom_kg.get("chunks", []):
                chunk_content = chunk_data["content"]
                source_id = chunk_data["source_id"]
                chunk_id = compute_mdhash_id(chunk_content.strip(), prefix="chunk-")

                chunk_entry = {"content": chunk_content.strip(), "source_id": source_id}
                all_chunks_data[chunk_id] = chunk_entry
                chunk_to_source_map[source_id] = chunk_id
                update_storage = True

            if self.chunks_vdb is not None and all_chunks_data:
                await self.chunks_vdb.upsert(all_chunks_data)
            if self.text_chunks is not None and all_chunks_data:
                await self.text_chunks.upsert(all_chunks_data)

            # Insert entities into knowledge graph
            all_entities_data = []
            for entity_data in custom_kg.get("entities", []):
                entity_name = f'"{entity_data["entity_name"].upper()}"'
                entity_type = entity_data.get("entity_type", "UNKNOWN")
                description = entity_data.get("description", "No description provided")
                source_chunk_id = entity_data.get("source_id", "UNKNOWN")
                source_id = chunk_to_source_map.get(source_chunk_id, "UNKNOWN")

                # Log if source_id is UNKNOWN
                if source_id == "UNKNOWN":
                    logger.warning(
                        f"Entity '{entity_name}' has an UNKNOWN source_id. Please check the source mapping."
                    )

                # Prepare node data
                node_data = {
                    "entity_type": entity_type,
                    "description": description,
                    "source_id": source_id,
                }
                # Insert node data into the knowledge graph
                await self.chunk_entity_relation_graph.upsert_node(
                    entity_name, node_data=node_data
                )
                node_data["entity_name"] = entity_name
                all_entities_data.append(node_data)
                update_storage = True

            # Insert relationships into knowledge graph
            all_relationships_data = []
            for relationship_data in custom_kg.get("relationships", []):
                src_id = f'"{relationship_data["src_id"].upper()}"'
                tgt_id = f'"{relationship_data["tgt_id"].upper()}"'
                description = relationship_data["description"]
                keywords = relationship_data["keywords"]
                weight = relationship_data.get("weight", 1.0)
                source_chunk_id = relationship_data.get("source_id", "UNKNOWN")
                source_id = chunk_to_source_map.get(source_chunk_id, "UNKNOWN")

                # Log if source_id is UNKNOWN
                if source_id == "UNKNOWN":
                    logger.warning(
                        f"Relationship from '{src_id}' to '{tgt_id}' has an UNKNOWN source_id. Please check the source mapping."
                    )

                # Check if nodes exist in the knowledge graph
                for need_insert_id in [src_id, tgt_id]:
                    if not (
                        await self.chunk_entity_relation_graph.has_node(need_insert_id)
                    ):
                        await self.chunk_entity_relation_graph.upsert_node(
                            need_insert_id,
                            node_data={
                                "source_id": source_id,
                                "description": "UNKNOWN",
                                "entity_type": "UNKNOWN",
                            },
                        )

                # Insert edge into the knowledge graph
                await self.chunk_entity_relation_graph.upsert_edge(
                    src_id,
                    tgt_id,
                    edge_data={
                        "weight": weight,
                        "description": description,
                        "keywords": keywords,
                        "source_id": source_id,
                    },
                )
                edge_data = {
                    "src_id": src_id,
                    "tgt_id": tgt_id,
                    "description": description,
                    "keywords": keywords,
                }
                all_relationships_data.append(edge_data)
                update_storage = True

            # Insert entities into vector storage if needed
            if self.entities_vdb is not None:
                data_for_vdb = {
                    compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                        "content": dp["entity_name"] + dp["description"],
                        "entity_name": dp["entity_name"],
                    }
                    for dp in all_entities_data
                }
                await self.entities_vdb.upsert(data_for_vdb)

            # Insert relationships into vector storage if needed
            if self.hyperedges_vdb is not None:
                data_for_vdb = {
                    compute_mdhash_id(dp["src_id"] + dp["tgt_id"], prefix="rel-"): {
                        "src_id": dp["src_id"],
                        "tgt_id": dp["tgt_id"],
                        "content": dp["keywords"]
                        + dp["src_id"]
                        + dp["tgt_id"]
                        + dp["description"],
                    }
                    for dp in all_relationships_data
                }
                await self.hyperedges_vdb.upsert(data_for_vdb)
        finally:
            if update_storage:
                await self._insert_done()

    def query(self, query: str, param: QueryParam = QueryParam(), entity_match=None, hyperedge_match=None):
        return self._run_sync(self.aquery(query, param, entity_match, hyperedge_match))

    @traced("aquery")
    async def aquery(self, query: str, param: QueryParam = QueryParam(), entity_match=None, hyperedge_match=None):
        inc_gauge("active_queries")
        inc_counter("aquery_total")
        t_start = __import__("time").time()
        try:
            if self._query_semaphore.locked():
                logger.warning(
                    f"[Overload] 查询信号量已耗尽 (上限={self.query_max_concurrent})，"
                    f"新请求将排队等待: {query[:100]}"
                )
            async with self._query_semaphore:
                # ----------------------------------------------------------------------
                # [增强] 自动检索逻辑 + [修改] 增加针对外部资源的硬性超时熔断
                # ----------------------------------------------------------------------
                if entity_match is None or hyperedge_match is None:
                    if self.entities_vdb and hasattr(self.entities_vdb, "query"):
                        logger.info(f"Auto-retrieving entities for query: {query}")
                        with trace_span("entity_retrieval"):
                            try:
                                # 使用 wait_for 防止向量库查询挂起引发协程雪崩
                                entity_match = await asyncio.wait_for(
                                    self.entities_vdb.query(query, top_k=param.top_k),
                                    timeout=self.db_timeout
                                )
                            except asyncio.TimeoutError:
                                logger.error(f"[Timeout] Entity retrieval timed out for query: {query}")
                                entity_match = []
                            except Exception as e:
                                logger.warning(f"Entity retrieval failed: {e}")
                                entity_match = []
                    else:
                        entity_match = []

                    if self.hyperedges_vdb and hasattr(self.hyperedges_vdb, "query"):
                        logger.info(f"Auto-retrieving hyperedges for query: {query}")
                        with trace_span("hyperedge_retrieval"):
                            try:
                                hyperedge_match = await asyncio.wait_for(
                                    self.hyperedges_vdb.query(query, top_k=param.top_k),
                                    timeout=self.db_timeout
                                )
                            except asyncio.TimeoutError:
                                logger.error(f"[Timeout] Hyperedge retrieval timed out for query: {query}")
                                hyperedge_match = []
                            except Exception as e:
                                logger.warning(f"Hyperedge retrieval failed: {e}")
                                hyperedge_match = []
                    else:
                        hyperedge_match = []
                # ----------------------------------------------------------------------

                if param.mode in ["hybrid"]:
                    try:
                        # 给知识图谱大模型推理节点加上全局超时保障
                        response = await asyncio.wait_for(
                            kg_query(
                                query,
                                self.chunk_entity_relation_graph,
                                entity_match,
                                hyperedge_match,
                                self.text_chunks,
                                param,
                                asdict(self),
                                hashing_kv=self.llm_response_cache,
                            ),
                            timeout=self.llm_timeout
                        )
                    except asyncio.TimeoutError:
                        logger.error(f"[Timeout] LLM Hybrid Query timed out for: {query}")
                        response = "System Error: Knowledge Graph query timed out."
                    except Exception as e:
                        logger.error(f"[Error] LLM Hybrid Query failed: {e}")
                        response = "System Error: Failed to generate response from Knowledge Graph."
                else:
                    response = "Only hybrid mode is supported currently."

                await self._query_done()
                observe_histogram("aquery_duration_seconds", __import__("time").time() - t_start)
                return response
        except Exception:
            inc_counter("aquery_errors_total")
            raise
        finally:
            dec_gauge("active_queries")

    async def _query_done(self):
        tasks = []
        for storage_inst in [self.llm_response_cache]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
            
        # [修改] return_exceptions=True 防异常穿透雪崩
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Task {idx} failed in _query_done with error: {result}")

    def delete_by_entity(self, entity_name: str):
        return self._run_sync(self.adelete_by_entity(entity_name))

    async def adelete_by_entity(self, entity_name: str):
        entity_name = f'"{entity_name.upper()}"'

        try:
            await self.entities_vdb.delete_entity(entity_name)
            await self.hyperedges_vdb.delete_relation(entity_name)
            await self.chunk_entity_relation_graph.delete_node(entity_name)

            logger.info(
                f"Entity '{entity_name}' and its relationships have been deleted."
            )
            await self._delete_by_entity_done()
        except Exception as e:
            logger.error(f"Error while deleting entity '{entity_name}': {e}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Hypergraph-aware cascading document deletion
    # ═══════════════════════════════════════════════════════════════════════════

    def delete_by_document(self, paper_name: str):
        """Synchronous wrapper for :meth:`adelete_by_document`."""
        return self._run_sync(self.adelete_by_document(paper_name))

    async def adelete_by_document(self, paper_name: str):
        """Delete a document and cascade-clean orphaned hypergraph structures.

        Each hyperedge belongs to exactly one document, so all hyperedges
        under *paper_name* are deleted together with it.  Entities, however,
        may be shared across hyperedges from different documents — they are
        only deleted when no surviving hyperedge references them.

        Steps:

        1. Find the document node.
        2. Collect all hyperedges connected to this document.
        3. For each hyperedge, collect the entity nodes it references.
        4. Classify entities:
           - **orphaned** (connected only to this documentʼs hyperedges) → delete.
           - **shared** (has at least one neighbour outside the deleted set) → keep,
             only remove the edge to the deleted hyperedge.
        5. Remove all paper→hyperedge edges.
        6. Delete orphaned entities (graph + vector DB).
        7. Remove shared-entity → hyperedge edges.
        8. Delete hyperedge nodes (graph + vector DB).
        9. Delete the document node.

        Parameters
        ----------
        paper_name : str
            The paper / document identifier used during :meth:`ainsert`.
        """
        kg = self.chunk_entity_relation_graph

        # ── Guard: document must exist ──────────────────────────────────
        if not await kg.has_node(paper_name):
            logger.warning(f"Document '{paper_name}' not found in graph; nothing to delete.")
            return

        # ── Phase 1: collect hyperedges directly under this document ────
        doc_edges = await kg.get_node_edges(paper_name)
        if not doc_edges:
            logger.info(f"Document '{paper_name}' has no hyperedges; deleting node only.")
            await kg.delete_node(paper_name)
            await self._delete_by_document_done()
            return

        # doc_edges: [(paper_name, neighbor), …] for undirected graph → neighbor = hyperedge
        doc_hyperedges: set[str] = set()
        for edge in doc_edges:
            neighbour = edge[1] if edge[0] == paper_name else edge[0]
            doc_hyperedges.add(neighbour)

        # ── Phase 2: for each hyperedge, collect entity neighbours ──────
        hyperedge_to_entities: dict[str, set[str]] = {}
        all_affected_entities: set[str] = set()

        for he in doc_hyperedges:
            he_edges = await kg.get_node_edges(he)
            if not he_edges:
                continue
            entities: set[str] = set()
            for edge in he_edges:
                neighbour = edge[1] if edge[0] == he else edge[0]
                if neighbour == paper_name:
                    continue  # skip the document→hyperedge back-edge
                node_data = await kg.get_node(neighbour)
                if node_data and node_data.get("role") == "entity":
                    entities.add(neighbour)
            if entities:
                hyperedge_to_entities[he] = entities
                all_affected_entities.update(entities)

        # ── Phase 3: classify entities ──────────────────────────────────
        # orphaned  = connected ONLY to hyperedges inside *doc_hyperedges*
        # shared    = has at least one neighbour outside the deleted set
        entities_to_delete: set[str] = set()
        entities_to_keep: set[str] = set()

        for entity in all_affected_entities:
            ent_edges = await kg.get_node_edges(entity)
            if not ent_edges:
                entities_to_delete.add(entity)
                continue

            surviving_neighbours: set[str] = set()
            for edge in ent_edges:
                neighbour = edge[1] if edge[0] == entity else edge[0]
                if neighbour not in doc_hyperedges and neighbour != paper_name:
                    surviving_neighbours.add(neighbour)

            if surviving_neighbours:
                entities_to_keep.add(entity)
            else:
                entities_to_delete.add(entity)

        logger.info(
            f"[Cascade] Document '{paper_name}': "
            f"{len(doc_hyperedges)} hyperedges, "
            f"{len(entities_to_delete)} entities to delete, "
            f"{len(entities_to_keep)} entities to keep (shared)."
        )

        # ── Phase 4: remove paper→hyperedge edges ───────────────────────
        for he in doc_hyperedges:
            if await kg.has_edge(paper_name, he):
                await kg.delete_edge(paper_name, he)

        # ── Phase 5: delete orphaned entities (graph + vector DB) ───────
        for entity in entities_to_delete:
            logger.info(f"[Cascade] Deleting orphaned entity: {entity}")
            try:
                await self.entities_vdb.delete_entity(entity)
            except Exception as e:
                logger.error(f"[Cascade] Vector-DB entity deletion failed for '{entity}': {e}")
            try:
                await self.hyperedges_vdb.delete_relation(entity)
            except Exception as e:
                logger.error(f"[Cascade] Vector-DB relation cleanup failed for '{entity}': {e}")
            try:
                await kg.delete_node(entity)
            except Exception as e:
                logger.error(f"[Cascade] Graph node deletion failed for '{entity}': {e}")

        # ── Phase 6: remove shared-entity → hyperedge edges (entity survives) ──
        for entity in entities_to_keep:
            for he in doc_hyperedges:
                if await kg.has_edge(he, entity):
                    logger.info(
                        f"[Cascade] Removing edge '{he}' → '{entity}' "
                        f"(entity shared across documents)."
                    )
                    await kg.delete_edge(he, entity)

        # ── Phase 7: delete hyperedge nodes (graph + vector DB) ─────────
        for he in doc_hyperedges:
            logger.info(f"[Cascade] Deleting hyperedge: {he}")
            try:
                await self.hyperedges_vdb.delete_by_hyperedge_name(he)
            except Exception as e:
                logger.error(f"[Cascade] Vector-DB hyperedge deletion failed for '{he}': {e}")
            try:
                await kg.delete_node(he)
            except Exception as e:
                logger.error(f"[Cascade] Graph node deletion failed for '{he}': {e}")

        # ── Phase 8: delete the document node itself ────────────────────
        await kg.delete_node(paper_name)

        logger.info(f"[Cascade] Document '{paper_name}' fully purged.")
        await self._delete_by_document_done()

    async def _delete_by_document_done(self):
        """Persist storage after document deletion."""
        tasks = []
        for storage_inst in [
            self.entities_vdb,
            self.hyperedges_vdb,
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Task {idx} failed in _delete_by_document_done: {result}")

    async def _delete_by_entity_done(self):
        tasks = []
        for storage_inst in [
            self.entities_vdb,
            self.hyperedges_vdb,
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
            
        # [修改] return_exceptions=True 防异常穿透雪崩
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Task {idx} failed in _delete_by_entity_done with error: {result}")