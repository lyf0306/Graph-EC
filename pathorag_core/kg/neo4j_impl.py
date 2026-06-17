# neo4j_impl.py
import os
import asyncio
import time
from typing import Optional
from neo4j import AsyncGraphDatabase
from neo4j.exceptions import Neo4jError, DriverError
from ..base import BaseGraphStorage
from ..instrumentation import inc_counter, observe_histogram

# --- 新增: Tenacity 智能重试库 ---
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    wait_random,
    retry_if_exception,
    before_sleep_log
)
import logging

# 定义日志记录器
logger = logging.getLogger(__name__)

NEO4J_READ_SEMAPHORE = asyncio.Semaphore(30)

def is_retryable_error(exception):
    """判断是否为临时的、可重试的 Neo4j 错误。"""
    if not isinstance(exception, (Neo4jError, DriverError)):
        return False

    code = getattr(exception, "code", "") or ""
    msg = getattr(exception, "message", "") or ""

    retryable_codes = [
        "ConstraintValidationFailed",
        "IndexEntryConflict",
        "DeadlockDetected",
        "LockClientStopped",
        "RateLimit",
        "TransientError",
        "ServiceUnavailable",
        "ConnectionReadTimeout"
    ]

    return any(c in code for c in retryable_codes) or "connection" in msg.lower()

READ_RETRY = dict(
    retry=retry_if_exception(is_retryable_error),
    wait=wait_exponential(multiplier=0.5, min=0.1, max=3) + wait_random(0, 0.3),
    stop=stop_after_attempt(3),
    reraise=True,
)

class Neo4JStorage(BaseGraphStorage):
    def __init__(self, namespace: str, global_config: dict, embedding_func):
        super().__init__(namespace, global_config, embedding_func)
        self.uri = os.environ.get("NEO4J_URI", "neo4j://localhost:7687")
        self.username = os.environ.get("NEO4J_USERNAME", "neo4j")
        self.password = os.environ.get("NEO4J_PASSWORD", "neo4j")
        self.database = os.environ.get("NEO4J_DATABASE", "neo4j")
        
        try:
            self.driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password),
                max_connection_pool_size=50,
                connection_acquisition_timeout=10,
            )
        except Exception as e:
            raise ConnectionError(
                f"Failed to connect to Neo4j. "
                f"Please check URI, username, and password in environment variables. Error: {e}"
            )

    async def close(self):
        await self.driver.close()

    # --- 修改 upsert_node: 添加 @retry 装饰器，内部逻辑保持不变 ---
    @retry(
        retry=retry_if_exception(is_retryable_error), # 只有特定的错误才重试
        wait=wait_exponential(multiplier=0.5, min=0.2, max=10) + wait_random(0, 0.5), # 指数退避+随机抖动
        stop=stop_after_attempt(10), # 最多重试10次
        reraise=True # 失败后抛出异常
    )
    async def upsert_node(self, node_name: str, node_data: dict):
        op_name = "upsert_node"
        start = time.time()
        inc_counter("neo4j_operations_total", labels={"operation": op_name})
        try:
            async with self.driver.session(database=self.database) as session:
                labels = []
                if "role" in node_data:
                    role_label = "".join(e for e in node_data["role"] if e.isalnum())
                    if role_label:
                        labels.append(role_label.capitalize())

                if "entity_type" in node_data:
                    type_label = "".join(e for e in node_data["entity_type"] if e.isalnum())
                    if type_label:
                        labels.append(type_label.capitalize())

                if not labels:
                    labels.append("Node")

                cypher_labels = ":" + ":".join(labels)

                props = {**node_data}
                props.pop("role", None)
                # props.pop("entity_type", None) <--- 保持这一行注释状态，保留 entity_type 属性
                props["name"] = node_name

                query = (
                    f"MERGE (n {{name: $name}}) "      # 仅按名称匹配/创建
                    f"SET n {cypher_labels} "          # 补全 Label (如 :Paper:Entity)
                    "SET n += $props"
                )
                await session.run(query, name=node_name, props=props)
            observe_histogram("neo4j_operation_duration_seconds", time.time() - start, labels={"operation": op_name})
        except Exception:
            inc_counter("neo4j_errors_total", labels={"operation": op_name})
            raise

    # --- 修改 upsert_edge: 添加 @retry 装饰器 ---
    @retry(
        retry=retry_if_exception(is_retryable_error),
        wait=wait_exponential(multiplier=0.5, min=0.2, max=10) + wait_random(0, 0.5),
        stop=stop_after_attempt(10),
        reraise=True
    )
    async def upsert_edge(
        self, src_node_name: str, tgt_node_name: str, edge_data: dict
    ):
        op_name = "upsert_edge"
        start = time.time()
        inc_counter("neo4j_operations_total", labels={"operation": op_name})
        try:
            async with self.driver.session(database=self.database) as session:
                # 默认类型
                rel_type = "RELATES_TO"

                # 如果 edge_data 中包含 description，尝试将其作为关系类型
                if "description" in edge_data:
                    candidate_type = edge_data["description"].upper()
                    # 清洗字符，确保只包含字母、数字和下划线，符合 Neo4j 命名规范
                    candidate_type = "".join(e for e in candidate_type if e.isalnum() or e == '_')

                    # 如果清洗后不为空，则使用该类型
                    if candidate_type:
                        rel_type = candidate_type

                props = {**edge_data} # 复制属性

                # 构建 Cypher 查询
                query = (
                    "MERGE (a {name: $src_name}) "
                    "MERGE (b {name: $tgt_name}) "
                    f"MERGE (a)-[r:{rel_type}]->(b) "
                    "SET r += $props"
                )

                await session.run(
                    query,
                    src_name=src_node_name,
                    tgt_name=tgt_node_name,
                    props=props
                )
            observe_histogram("neo4j_operation_duration_seconds", time.time() - start, labels={"operation": op_name})
        except Exception:
            inc_counter("neo4j_errors_total", labels={"operation": op_name})
            raise

    @retry(**READ_RETRY)
    async def get_node(self, node_name: str):
        op_name = "get_node"
        start = time.time()
        inc_counter("neo4j_operations_total", labels={"operation": op_name})
        try:
            async with self.driver.session(database=self.database) as session:
                query = "MATCH (n {name: $name}) RETURN n"
                result = await session.run(query, name=node_name)
                record = await result.single()
                # 保持简单的容错，防止意外
                if record:
                    data = dict(record[0])
                    # 如果数据库中真的因为早期版本缺失了 entity_type，给个默认值防止报错
                    if "entity_type" not in data:
                        data["entity_type"] = "UNKNOWN"
                    observe_histogram("neo4j_operation_duration_seconds", time.time() - start, labels={"operation": op_name})
                    return data
                observe_histogram("neo4j_operation_duration_seconds", time.time() - start, labels={"operation": op_name})
                return None
        except Exception:
            inc_counter("neo4j_errors_total", labels={"operation": op_name})
            raise

    # 请在你的 Neo4j 实现类中加入这个方法
    @retry(**READ_RETRY)
    async def get_node_edges_with_roles(self, source_node_name: str):
        op_name = "get_node_edges_with_roles"
        start = time.time()
        inc_counter("neo4j_operations_total", labels={"operation": op_name})
        try:
            if not await self.has_node(source_node_name):
                observe_histogram("neo4j_operation_duration_seconds", time.time() - start, labels={"operation": op_name})
                return []
            async with self.driver.session(database=self.database) as session:
                # 【关键修改】：加上 ORDER BY 强制按名称排序，保证每次查出来的边顺序绝对一致
                query = "MATCH (n {name: $name})-[r]-(m) RETURN m.name as target, r.role as role ORDER BY m.name ASC"
                result = await session.run(query, name=source_node_name)
                edges = []
                async for record in result:
                    edges.append((source_node_name, record["target"], record.get("role", "UNKNOWN")))
                observe_histogram("neo4j_operation_duration_seconds", time.time() - start, labels={"operation": op_name})
                return edges
        except Exception:
            inc_counter("neo4j_errors_total", labels={"operation": op_name})
            raise

    @retry(**READ_RETRY)
    async def get_edge(self, src_node_name: str, tgt_node_name: str):
        op_name = "get_edge"
        start = time.time()
        inc_counter("neo4j_operations_total", labels={"operation": op_name})
        try:
            async with self.driver.session(database=self.database) as session:
                query = "MATCH (a {name: $src_name})-[r]->(b {name: $tgt_name}) RETURN r"
                result = await session.run(
                    query, src_name=src_node_name, tgt_name=tgt_node_name
                )
                record = await result.single()
                result_val = dict(record[0]) if record else None
                observe_histogram("neo4j_operation_duration_seconds", time.time() - start, labels={"operation": op_name})
                return result_val
        except Exception:
            inc_counter("neo4j_errors_total", labels={"operation": op_name})
            raise

    @retry(**READ_RETRY)
    async def has_node(self, node_name: str) -> bool:
        op_name = "has_node"
        start = time.time()
        inc_counter("neo4j_operations_total", labels={"operation": op_name})
        try:
            async with self.driver.session(database=self.database) as session:
                query = "MATCH (n {name: $name}) RETURN count(n) as count"
                result = await session.run(query, name=node_name)
                record = await result.single()
                result_val = record["count"] > 0
                observe_histogram("neo4j_operation_duration_seconds", time.time() - start, labels={"operation": op_name})
                return result_val
        except Exception:
            inc_counter("neo4j_errors_total", labels={"operation": op_name})
            raise

    @retry(**READ_RETRY)
    async def has_edge(self, src_node_name: str, tgt_node_name: str) -> bool:
        op_name = "has_edge"
        start = time.time()
        inc_counter("neo4j_operations_total", labels={"operation": op_name})
        try:
            async with self.driver.session(database=self.database) as session:
                query = (
                    "MATCH (a {name: $src_name})-[r]->(b {name: $tgt_name}) "
                    "RETURN count(r) as count"
                )
                result = await session.run(
                    query, src_name=src_node_name, tgt_name=tgt_node_name
                )
                record = await result.single()
                result_val = record["count"] > 0
                observe_histogram("neo4j_operation_duration_seconds", time.time() - start, labels={"operation": op_name})
                return result_val
        except Exception:
            inc_counter("neo4j_errors_total", labels={"operation": op_name})
            raise

    @retry(**READ_RETRY)
    async def node_degree(self, node_name: str) -> int:
        """获取节点的度（连接数）"""
        op_name = "node_degree"
        start = time.time()
        inc_counter("neo4j_operations_total", labels={"operation": op_name})
        try:
            async with self.driver.session(database=self.database) as session:
                query = "MATCH (n {name: $name})-[r]-() RETURN count(r) as degree"
                result = await session.run(query, name=node_name)
                record = await result.single()
                result_val = record["degree"] if record else 0
                observe_histogram("neo4j_operation_duration_seconds", time.time() - start, labels={"operation": op_name})
                return result_val
        except Exception:
            inc_counter("neo4j_errors_total", labels={"operation": op_name})
            raise

    @retry(**READ_RETRY)
    async def edge_degree(self, src_node_name: str, tgt_node_name: str) -> int:
        """获取边的度（定义为源节点度 + 目标节点度）"""
        op_name = "edge_degree"
        start = time.time()
        inc_counter("neo4j_operations_total", labels={"operation": op_name})
        try:
            async with self.driver.session(database=self.database) as session:
                # 修复逻辑：分步计算 d1 和 d2，最后再相加
                query = """
                    MATCH (src {name: $src_name})
                    OPTIONAL MATCH (src)-[r1]-()
                    WITH count(r1) as d1
                    MATCH (tgt {name: $tgt_name})
                    OPTIONAL MATCH (tgt)-[r2]-()
                    WITH d1, count(r2) as d2
                    RETURN d1 + d2 as degree
                """
                result = await session.run(
                    query, src_name=src_node_name, tgt_name=tgt_node_name
                )
                record = await result.single()
                result_val = record["degree"] if record else 0
                observe_histogram("neo4j_operation_duration_seconds", time.time() - start, labels={"operation": op_name})
                return result_val
        except Exception:
            inc_counter("neo4j_errors_total", labels={"operation": op_name})
            raise

    @retry(**READ_RETRY)
    async def get_node_edges(self, source_node_name: str):
        op_name = "get_node_edges"
        start = time.time()
        inc_counter("neo4j_operations_total", labels={"operation": op_name})
        try:
            if not await self.has_node(source_node_name):
                observe_histogram("neo4j_operation_duration_seconds", time.time() - start, labels={"operation": op_name})
                return None
            async with self.driver.session(database=self.database) as session:
                # 【关键修改】：同样加上 ORDER BY
                query = "MATCH (n {name: $name})-[r]-(m) RETURN m.name as target ORDER BY m.name ASC"
                result = await session.run(query, name=source_node_name)
                edges = []
                async for record in result:
                    edges.append((source_node_name, record["target"]))
                observe_histogram("neo4j_operation_duration_seconds", time.time() - start, labels={"operation": op_name})
                return edges
        except Exception:
            inc_counter("neo4j_errors_total", labels={"operation": op_name})
            raise

    async def index_done_callback(self):
        await asyncio.sleep(0.0)
        return

    async def delete_node(self, node_name: str):
        op_name = "delete_node"
        start = time.time()
        inc_counter("neo4j_operations_total", labels={"operation": op_name})
        try:
            async with self.driver.session(database=self.database) as session:
                query = "MATCH (n {name: $name}) DETACH DELETE n"
                await session.run(query, name=node_name)
            observe_histogram("neo4j_operation_duration_seconds", time.time() - start, labels={"operation": op_name})
        except Exception:
            inc_counter("neo4j_errors_total", labels={"operation": op_name})
            raise

    async def delete_edge(self, source_node_id: str, target_node_id: str):
        op_name = "delete_edge"
        start = time.time()
        inc_counter("neo4j_operations_total", labels={"operation": op_name})
        try:
            async with self.driver.session(database=self.database) as session:
                query = (
                    "MATCH (a {name: $src})-[r]->(b {name: $tgt}) DELETE r"
                )
                await session.run(query, src=source_node_id, tgt=target_node_id)
            observe_histogram("neo4j_operation_duration_seconds", time.time() - start, labels={"operation": op_name})
        except Exception:
            inc_counter("neo4j_errors_total", labels={"operation": op_name}")
            raise

    @retry(**READ_RETRY)
    async def get_paper_by_pmid(self, pmid: str)  -> Optional[str]:
        op_name = "get_paper_by_pmid"
        start = time.time()
        inc_counter("neo4j_operations_total", labels={"operation": op_name})
        try:
            async with self.driver.session(database=self.database) as session:
                query = """
                        MATCH (p:Paper {pmid: $pmid})
                        RETURN p.name AS name
                        """
                result = await session.run(query, pmid=pmid)
                record = await result.single()
                result_val = record["name"] if record else None
                observe_histogram("neo4j_operation_duration_seconds", time.time() - start, labels={"operation": op_name})
                return result_val
        except Exception:
            inc_counter("neo4j_errors_total", labels={"operation": op_name})
            raise

    async def update_paper_guidelines(self, pmid: str, new_guideline: str):
        op_name = "update_paper_guidelines"
        start = time.time()
        inc_counter("neo4j_operations_total", labels={"operation": op_name})
        try:
            async with self.driver.session(database=self.database) as session:
                # 使用 apoc 或简单的 SET 逻辑确保指南不重复
                query = """
                MATCH (n:Paper {pmid: $pmid})
                SET n.guidelines = CASE
                    WHEN $new_g IN n.guidelines THEN n.guidelines
                    ELSE n.guidelines + $new_g
                END
                """
                await session.run(query, pmid=pmid, new_g=new_guideline)
            observe_histogram("neo4j_operation_duration_seconds", time.time() - start, labels={"operation": op_name})
        except Exception:
            inc_counter("neo4j_errors_total", labels={"operation": op_name})
            raise