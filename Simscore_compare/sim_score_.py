from neo4j import GraphDatabase
import pandas as pd
import os

url=os.getenv("NEO4J_URL", "neo4j://127.0.0.1:7687")
username=os.getenv("NEO4J_USERNAME", "neo4j")
password=os.getenv("NEO4J_PASSWORD", "12345678")

driver = GraphDatabase.driver(url, auth=(username, password))

#%%
CUTOFF_CSV = r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\Simscore\2020_cutoffs.csv"
OUTPUT_CSV = r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\Simscore\results\2020_.csv"
df = pd.read_csv(CUTOFF_CSV, encoding="utf-8-sig")


dates = df.columns[1:]  # 날짜 컬럼
percents = df["Percent"]

INDEX_START = 0
INDEX_END = INDEX_START + len(dates) - 1

def run_query(driver, gid, cutoff, index, cutoff_2):
    query = f"""
    MATCH (n)
    WHERE n.gid = '{gid}' AND NOT n:Summary

    MATCH (n)-[r:reference]->(m)
    WHERE NOT m:Summary

    MATCH (m)-[s]-(o)
    WHERE NOT o:Summary 
      AND TYPE(s) <> 'reference' 
      AND o.sim_score[{index}] > {cutoff}
      
    RETURN count(DISTINCT n) + count(DISTINCT m) + count(DISTINCT o) AS node_count,
           count(DISTINCT r) + count(DISTINCT s) AS rel_count
    """
    with driver.session() as session:
        result = session.run(query)
        return result.data()[0]

results = []
for offset, date in enumerate(dates):
    sim_index = INDEX_START + offset   # ★ 자동으로 sim_score 인덱스 계산
    gid = f"FOMC{date}"

    for i, p in enumerate(percents):
        cutoff = df.loc[i, date]
        record = run_query(driver, gid, cutoff, sim_index)
        results.append({
            "Date": date,
            "Percent": p,
            "SimScoreIndex": sim_index,
            "Cutoff": cutoff,
            "NodeCount": record["node_count"],
            "RelCount": record["rel_count"]
        })

# 6) 결과 CSV 저장
result_df = pd.DataFrame(results)
result_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print(f"[완료] 결과 저장 → {OUTPUT_CSV}")
print(f"처리한 sim_score 인덱스 범위: {INDEX_START} ~ {INDEX_END}")