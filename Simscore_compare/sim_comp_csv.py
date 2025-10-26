from neo4j import GraphDatabase
import pandas as pd
import os

url=os.getenv("NEO4J_URL", "neo4j://127.0.0.1:7687")
username=os.getenv("NEO4J_USERNAME", "neo4j")
password=os.getenv("NEO4J_PASSWORD", "12345678")

driver = GraphDatabase.driver(url, auth=(username, password))
#%%

query = """
MATCH (n)
WHERE n.gid = 'FOMC201910' AND NOT n:Summary

MATCH (n)-[r:reference]->(m)
WHERE NOT m:Summary

MATCH (m)-[s]-(o)
WHERE NOT o:Summary 
  AND TYPE(s) <> 'reference' 
  AND o.sim_score[22] > 0.622569318
 

RETURN count(DISTINCT n) + count(DISTINCT m) + count(DISTINCT o) AS node_count,
       count(DISTINCT r) + count(DISTINCT s) AS rel_count
"""
def run_query(driver, query):
    with driver.session() as session:
        result = session.run(query)
        return result.data()

records = run_query(driver, query)
#%%
print(f"노드 개수: {records[0]['node_count']}")
print(f"관계 개수: {records[0]['rel_count']}")
#AND o.sim_score[26] > 0.663596289