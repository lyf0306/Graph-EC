import math
import os
from neo4j import GraphDatabase
from tqdm import tqdm

# ================= 配置区域 =================
NEO4J_URI = "neo4j://localhost:7688"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")

# 权重平滑基数 (防止极端低频词权重过大爆炸，可微调)
SMOOTHING_BASE = 1.0 
# ===========================================

class Neo4jIDFCalculator:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def calculate_and_update_idf(self):
        with self.driver.session() as session:
            print(">>> [1/4] 正在统计图谱中的总超边数量 (Total Hyperedges/N)...")
            # 查询所有带有 <hyperedge> 前缀的节点总数
            result_n = session.run("""
                MATCH (h) 
                WHERE h.name STARTS WITH '<hyperedge>' 
                RETURN count(h) AS total_N
            """)
            total_N = result_n.single()["total_N"]
            print(f"  └─ 发现总超边数量 (N) = {total_N}")

            if total_N == 0:
                print("❌ 错误：图谱中没有找到超边，请检查数据是否已正确导入！")
                return

            print("\n>>> [2/4] 正在统计各实体节点的文档频率 (Document Frequency/df)...")
            # 查询每个实体节点连接了多少个不同的超边
            # 这里的 (e)-[:RELATES_TO]->(h) 对应 test.py 里的拓扑结构
            result_df = session.run("""
                MATCH (e)-[:RELATES_TO]->(h)
                WHERE h.name STARTS WITH '<hyperedge>' 
                  AND NOT e.name STARTS WITH '<hyperedge>'
                  AND NOT e:Paper
                RETURN e.name AS entity_name, count(DISTINCT h) AS df
            """)
            
            records = list(result_df)
            total_entities = len(records)
            print(f"  └─ 共提取到 {total_entities} 个有效实体节点。")

            print("\n>>> [3/4] 正在计算临床信息熵 (IDF Weights)...")
            update_batch = []
            
            # 记录最高和最低权重的词，方便控制台展示和校验
            stats_list = []

            for record in records:
                entity_name = record["entity_name"]
                df = record["df"]
                
                # 核心公式：IDF = log(N / (df + 1)) + 1.0
                # +1 是为了防止分母为0，外面的 +1.0 是基础权重保底
                idf_weight = math.log(total_N / (df + 1)) + SMOOTHING_BASE
                
                # 保留4位小数
                idf_weight = round(idf_weight, 4)
                
                update_batch.append({
                    "entity_name": entity_name,
                    "idf_weight": idf_weight,
                    "df": df
                })
                stats_list.append((entity_name, idf_weight, df))

            # 打印 Top 5 罕见/致命特征 和 Top 5 常见废话特征
            stats_list.sort(key=lambda x: x[1], reverse=True)
            print("\n  [📊 权重分布探针]")
            print("  ⭐ Top 5 高权重实体 (罕见/特异性极强):")
            for name, w, df in stats_list[:5]:
                print(f"     - {name[:30]:<30} | 权重: {w:.4f} | 出现频次: {df}")
            
            print("  🔻 Top 5 低权重实体 (常见/通用型描述):")
            for name, w, df in stats_list[-5:]:
                print(f"     - {name[:30]:<30} | 权重: {w:.4f} | 出现频次: {df}")

            print(f"\n>>> [4/4] 正在将 {len(update_batch)} 个权重数据批量写回 Neo4j 数据库...")
            
            # 使用 UNWIND 进行高效的批量更新
            update_query = """
            UNWIND $batch AS data
            MATCH (e) WHERE e.name = data.entity_name
            SET e.idf_weight = data.idf_weight
            """
            
            # 分批写入，防止内存溢出 (每批 1000 个)
            batch_size = 1000
            for i in tqdm(range(0, len(update_batch), batch_size), desc="写入进度"):
                current_batch = update_batch[i:i + batch_size]
                session.run(update_query, batch=current_batch)

            print("\n✅ 所有实体 IDF 权重已成功写入 Neo4j！")
            print("现在你可以去 operate.py 里修改饱和度公式了！")

if __name__ == "__main__":
    calculator = Neo4jIDFCalculator(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
    try:
        calculator.calculate_and_update_idf()
    finally:
        calculator.close()