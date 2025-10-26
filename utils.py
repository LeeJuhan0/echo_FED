from openai import OpenAI
import os
from neo4j import GraphDatabase
import numpy as np
from camel.storages import Neo4jGraph
import uuid
from summerize import process_chunks
import openai
import re
import csv
sys_prompt_one = """
Rate the economic sentiment of the following text on a scale from -1.00 (very negative) to 1.00 (very positive). 
Provide the score rounded to two decimal places.
Then, identify and quote the single most important sentence that supports your rating. If no such sentence exists, write: “There is no basis.”

Finally, present the result with the sentiment score in this exact format at the end of the text: "sentiment score = %float"
"""

sys_prompt_two = """
Modify the response to the question by incorporating insights from provided financial books or academic papers. 
Should adjust the sentiment score accordingly, ensuring that the assessment reflects provided literature in economics and finance.

The output must end with the sentiment score in the following exact format: "sentiment score = %float"
"""

sys_prompt_three = """
Modify the response to the question using the provided Financial Stability Report (FSR) as a reference. 
Should adjust the sentiment score based on insights from the FSR.

The final output must end with the sentiment score in the exact format: "sentiment score = %float"
"""

sys_prompt_four = """
Modify the response to the question using the provided Senior Loan Officer Opinion Survey (SLOOS) as a reference. 
Should adjust the sentiment score based on insights from the SLOOS.

The final output must end with the sentiment score in the exact format: "sentiment score = %float"
"""

sys_prompt_five = """
Modify the response to the question using the provided Beige Book as a reference. 
Should adjust the sentiment score based on insights from the Beige Book.

The final output must end with the sentiment score in the exact format: "sentiment score = %float"
"""

# Add your own OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")

def save_responses_per_meeting(year: int, meeting_no: int, responses):
    """
    특정 연도(year)-회의번호(meeting_no) 폴더에
    responses 리스트(예: 5개)를 1.csv ~ N.csv로 저장.
    """
    BASE_PATH = r"C:/Users/HUFS_MATH/IdeaProjects/FOMC_Graphrag/Simulation_19000"
    folder_name = f"{year}-{meeting_no}"
    folder_path = os.path.join(BASE_PATH, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    for idx, response in enumerate(responses, start=1):
        file_path = os.path.join(folder_path, f"{idx}.csv")
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Response"])
            writer.writerow([response])

def get_embedding(text, mod = "text-embedding-3-small"):
    client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))

    response = client.embeddings.create(
        input=text,
        model=mod
    )

    return response.data[0].embedding

def add_text_property(graph_element, text_value: str, field: str = "source_text"):
    for node in graph_element.nodes:
        node.properties[field] = text_value
    return graph_element

def fetch_texts(n4j):
    # Fetch the text for each node
    query = "MATCH (n) RETURN n.id AS id"
    return n4j.query(query)

def add_embeddings(n4j, node_id, embedding):
    # Upload embeddings to Neo4j
    query = "MATCH (n) WHERE n.id = $node_id SET n.embedding = $embedding"
    n4j.query(query, params = {"node_id":node_id, "embedding":embedding})

def add_nodes_emb(n4j):
    nodes = fetch_texts(n4j)

    for node in nodes:
        # Calculate embedding for each node's text
        if node['id']:  # Ensure there is text to process
            embedding = get_embedding(node['id'])
            # Store embedding back in the node
            add_embeddings(n4j, node['id'], embedding)

def add_ge_emb(graph_element):
    for node in graph_element.nodes:
        emb = get_embedding(node.id)
        node.properties['embedding'] = emb
    return graph_element

def add_sim_score(graph_element, sim_score):
    for node in graph_element.nodes:
        node.properties['sim_score'] = sim_score
    for rel in graph_element.relationships:
        rel.properties['sim_score'] = sim_score
    return graph_element

def add_gid(graph_element, gid):
    for node in graph_element.nodes:
        node.properties['gid'] = gid
    for rel in graph_element.relationships:
        rel.properties['gid'] = gid
    return graph_element

def add_sum(n4j,content,gid):
    sum = process_chunks(content)
    creat_sum_query = """
        CREATE (s:Summary {content: $sum, gid: $gid})
        RETURN s
        """
    s = n4j.query(creat_sum_query, {'sum': sum, 'gid': gid})
    
    link_sum_query = """
        MATCH (s:Summary {gid: $gid}), (n)
        WHERE n.gid = s.gid AND NOT n:Summary
        CREATE (s)-[:SUMMARIZES]->(n)
        RETURN s, n
        """
    n4j.query(link_sum_query, {'gid': gid})

    return s

def call_llm(sys, user):
    response = openai.chat.completions.create(
        model="gpt-5",
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": f" {user}"},
        ],
        max_completion_tokens=3000, #for : gpt-5
        temperature=1,
        #max_tokens=3000, #for : gpt-4o
        #temperature=0.2,
        n=1,
        stop=None,
    )
    return response.choices[0].message.content

def find_index_of_largest(nums):
    # Sorting the list while keeping track of the original indexes
    sorted_with_index = sorted((num, index) for index, num in enumerate(nums))
    
    # Extracting the original index of the largest element
    largest_original_index = sorted_with_index[-1][1]
    
    return largest_original_index

def parsing_score(res):
    # 정규식 패턴: sentiment score = [공백][+-][공백]float
    pattern = r"sentiment score\s*=\s*([+-]?\s*[0-9]*\.[0-9]+)"
    match = re.search(pattern, res, re.IGNORECASE)
    if match:
        value = match.group(1).replace(' ', '')  # 공백 제거
        return float(value)
    else:
        print()
        return None


def get_response(n4j, gid, query, i, sim_score_median, year, meeting_no):
    responses = []
    score_list =[]
    selfcont = ret_context(n4j, gid)
    linkcont = link_context(n4j, gid, i, sim_score_median)
    linkcontFSR = link_context_FSR(n4j, gid)
    linkcontSLOOS = link_context_SLOOS(n4j, gid)
    linkcontbeigebook = link_context_beigebook(n4j, gid)

    user_one = "the question is: " + query
    res = call_llm(sys_prompt_one,user_one)
    responses.append(res)
    score_list.append(parsing_score(res))
    print("Summary의 대답 : " + res + "the references are: " +  "".join(selfcont))
    #
    user_two = "the question is: " + query + "the last response of it is: " +  res + "the references are: " +  "".join(linkcont)
    res = call_llm(sys_prompt_two,user_two)
    responses.append(res)
    score_list.append(parsing_score(res))
    print("paper의 대답 : "  + res)
    #
    user_three = "the question is:" + query + "the provided information is: " + res + "the references are: " + "".join(linkcontFSR)
    res = call_llm(sys_prompt_three,user_three)
    responses.append(res)
    score_list.append(parsing_score(res))
    print("FSR의 대답 : " + res)
    #
    user_four = "the question is:" + query + "the provided information is: " + res + "the references are: " + "".join(linkcontSLOOS)
    res = call_llm(sys_prompt_four, user_four)
    responses.append(res)
    score_list.append(parsing_score(res))
    print("SLOOS의 대답 : " + res)
    #
    user_five = "the question is:" + query + "the provided information is: " + res + "the references are: " + "".join(linkcontbeigebook)
    res = call_llm(sys_prompt_five, user_five)
    responses.append(res)
    score_list.append(parsing_score(res))
    print("beigebook의 대답 : " + res)
    save_responses_per_meeting(year, meeting_no, responses)
    return score_list

def link_context(n4j, gid, i, sim_score_median): #수정필요
    cont = []
    retrieve_query = """
        // Match all 'n' nodes with a specific gid but not of the "Summary" type
        MATCH (n)
        WHERE n.gid = $gid AND NOT n:Summary

        // Find all 'm' nodes where 'm' is a reference of 'n' via a 'REFERENCES' relationship
        MATCH (n)-[r:paragraph]->(m)
        WHERE NOT m:Summary

        // Find all 'o' nodes connected to each 'm', and include the relationship type,
        // while excluding 'Summary' type nodes and 'REFERENCE' relationship
        MATCH (m)-[s]-(o)
        WHERE NOT o:Summary AND TYPE(s) <> 'paragraph' AND o.sim_score[$i] > $sim_score_median 

        // Collect and return details in a structured format
        RETURN n.id AS NodeId1, 
            m.id AS Mid, 
            TYPE(r) AS paragraphType, 
            collect(DISTINCT {RelationType: type(s), Oid: o.id}) AS Connections
    """
    res = n4j.query(retrieve_query, {'gid': gid, 'i': i, 'sim_score_median': sim_score_median})
    for r in res:
        # Expand each set of connections into separate entries with n and m
        for ind, connection in enumerate(r["Connections"]):
            cont.append("Reference " + str(ind) + ": " + r["NodeId1"] + "has the reference that" + r['Mid'] + connection['RelationType'] + connection['Oid'])
    return cont

def link_context_FSR(n4j, gid):
    cont = []
    retrieve_query = """
        // Match all 'n' nodes with a specific gid but not of the "Summary" type
            MATCH (n)
        WHERE n.gid = $gid AND NOT n:Summary

        // Find all 'm' nodes where 'm' is a FSR of 'n' via a 'FSR' relationship
        MATCH (n)-[r:FSR]->(m)
        WHERE NOT m:Summary

        // Find all 'o' nodes connected to each 'm', and include the relationship type,
        // while excluding 'Summary' type nodes and 'FSR' relationship
        MATCH (m)-[s]-(o)
        WHERE NOT o:Summary AND TYPE(s) <> 'FSR'

        // Collect and return details in a structured format
        RETURN n.id AS NodeId1, 
            m.id AS Mid, 
            TYPE(r) AS FSRType, 
            collect(DISTINCT {RelationType: type(s), Oid: o.id}) AS Connections
    """
    res = n4j.query(retrieve_query, {'gid': gid})
    for r in res:
        # Expand each set of connections into separate entries with n and m
        for ind, connection in enumerate(r["Connections"]):
            cont.append("Reference " + str(ind) + ": " + r["NodeId1"] + "has the reference that" + r['Mid'] + connection['RelationType'] + connection['Oid'])
    return cont

def link_context_SLOOS(n4j, gid):
    cont = []
    retrieve_query = """
        // Match all 'n' nodes with a specific gid but not of the "Summary" type
        MATCH (n)
        WHERE n.gid = $gid AND NOT n:Summary

        // Find all 'm' nodes where 'm' is a reference of 'n' via a 'SLOOS' relationship
        MATCH (n)-[r:SLOOS]->(m)
        WHERE NOT m:Summary

        // Find all 'o' nodes connected to each 'm', and include the relationship type,
        // while excluding 'Summary' type nodes and 'SLOOS' relationship
        MATCH (m)-[s]-(o)
        WHERE NOT o:Summary AND TYPE(s) <> 'SLOOS'

        // Collect and return details in a structured format
        RETURN n.id AS NodeId1, 
            m.id AS Mid, 
            TYPE(r) AS SLOOSType, 
            collect(DISTINCT {RelationType: type(s), Oid: o.id}) AS Connections
    """
    res = n4j.query(retrieve_query, {'gid': gid})
    for r in res:
        # Expand each set of connections into separate entries with n and m
        for ind, connection in enumerate(r["Connections"]):
            cont.append("Reference " + str(ind) + ": " + r["NodeId1"] + "has the reference that" + r['Mid'] + connection['RelationType'] + connection['Oid'])
    return cont

def link_context_beigebook(n4j, gid):
    cont = []
    retrieve_query = """
        // Match all 'n' nodes with a specific gid but not of the "Summary" type
        MATCH (n)
        WHERE n.gid = $gid AND NOT n:Summary

        // Find all 'm' nodes where 'm' is a reference of 'n' via a 'beigebook' relationship
        MATCH (n)-[r:beigebook]->(m)
        WHERE NOT m:Summary

        // Find all 'o' nodes connected to each 'm', and include the relationship type,
        // while excluding 'Summary' type nodes and 'beigebook' relationship
        MATCH (m)-[s]-(o)
        WHERE NOT o:Summary AND TYPE(s) <> 'beigebook'

        // Collect and return details in a structured format
        RETURN n.id AS NodeId1, 
            m.id AS Mid, 
            TYPE(r) AS beigebookType, 
            collect(DISTINCT {RelationType: type(s), Oid: o.id}) AS Connections
    """
    res = n4j.query(retrieve_query, {'gid': gid})
    for r in res:
        # Expand each set of connections into separate entries with n and m
        for ind, connection in enumerate(r["Connections"]):
            cont.append("Reference " + str(ind) + ": " + r["NodeId1"] + "has the reference that" + r['Mid'] + connection['RelationType'] + connection['Oid'])
    return cont

def ret_context(n4j, gid):
    cont = []
    ret_query = """
    // Match all nodes with a specific gid but not of type "Summary" and collect them
    MATCH (n)
    WHERE n.gid = $gid AND NOT n:Summary
    WITH collect(n) AS nodes

    // Unwind the nodes to a pairs and match relationships between them
    UNWIND nodes AS n
    UNWIND nodes AS m
    MATCH (n)-[r]-(m)
    WHERE n.gid = m.gid AND id(n) < id(m) AND NOT n:Summary AND NOT m:Summary // Ensure each pair is processed once and exclude "Summary" nodes in relationships
    WITH n, m, TYPE(r) AS relType

    // Return node IDs and relationship types in structured format
    RETURN n.id AS NodeId1, relType, m.id AS NodeId2
    """
    res = n4j.query(ret_query, {'gid': gid})
    for r in res:
        cont.append(r['NodeId1'] + r['relType'] + r['NodeId2'])
    return cont

def merge_similar_nodes(n4j, gid):
    # Define your merge query here. Adjust labels and properties according to your graph schema
    if gid:
        merge_query = """
            WITH 0.5 AS threshold
            MATCH (n), (m)
            WHERE NOT n:Summary AND NOT m:Summary AND n.gid = m.gid AND n.gid = $gid AND n<>m AND apoc.coll.sort(labels(n)) = apoc.coll.sort(labels(m))
            WITH n, m,
                gds.similarity.cosine(n.embedding, m.embedding) AS similarity
            WHERE similarity > threshold
            WITH head(collect([n,m])) as nodes
            CALL apoc.refactor.mergeNodes(nodes, {properties: 'overwrite', mergeRels: true})
            YIELD node
            RETURN count(*)
        """
        result = n4j.query(merge_query, {'gid': gid})
    else:
        merge_query = """
            // Define a threshold for cosine similarity
            WITH 0.5 AS threshold
            MATCH (n), (m)
            WHERE NOT n:Summary AND NOT m:Summary AND n<>m AND apoc.coll.sort(labels(n)) = apoc.coll.sort(labels(m))
            WITH n, m,
                gds.similarity.cosine(n.embedding, m.embedding) AS similarity
            WHERE similarity > threshold
            WITH head(collect([n,m])) as nodes
            CALL apoc.refactor.mergeNodes(nodes, {properties: 'overwrite', mergeRels: true})
            YIELD node
            RETURN count(*)
        """
        result = n4j.query(merge_query)
    return result

def ref_link(n4j, gid1, gid2):
    trinity_query = """
        // Match nodes from Graph A
        MATCH (a)
        WHERE a.gid = $gid1 AND NOT a:Summary
        WITH collect(a) AS GraphA

        // Match nodes from Graph B
        MATCH (b)
        WHERE b.gid = $gid2 AND NOT b:Summary
        WITH GraphA, collect(b) AS GraphB

        // Unwind the nodes to compare each against each
        UNWIND GraphA AS n
        UNWIND GraphB AS m

        // Set the threshold for cosine similarity
        WITH n, m, 0.5 AS threshold

        // Compute cosine similarity and apply the threshold
        WHERE apoc.coll.sort(labels(n)) = apoc.coll.sort(labels(m)) AND n <> m
        WITH n, m, threshold,
            gds.similarity.cosine(n.embedding, m.embedding) AS similarity
        WHERE similarity > threshold

        // Create a relationship based on the condition
        MERGE (m)-[:paragraph]->(n)

        // Return results
        RETURN n, m
"""
    result = n4j.query(trinity_query, {'gid1': gid1, 'gid2': gid2})
    return result

def ref_link_FSR(n4j, gid1, gid2):
    trinity_query = """
        // Match nodes from Graph A
        MATCH (a)
        WHERE a.gid = $gid1 AND NOT a:Summary
        WITH collect(a) AS GraphA

        // Match nodes from Graph B
        MATCH (b)
        WHERE b.gid = $gid2 AND NOT b:Summary
        WITH GraphA, collect(b) AS GraphB

        // Unwind the nodes to compare each against each
        UNWIND GraphA AS n
        UNWIND GraphB AS m

        // Set the threshold for cosine similarity
        WITH n, m, 0.5 AS threshold

        // Compute cosine similarity and apply the threshold
        WHERE apoc.coll.sort(labels(n)) = apoc.coll.sort(labels(m)) AND n <> m
        WITH n, m, threshold,
            gds.similarity.cosine(n.embedding, m.embedding) AS similarity
        WHERE similarity > threshold

        // Create a relationship based on the condition
        MERGE (m)-[:FSR]->(n)

        // Return results
        RETURN n, m
"""


    result = n4j.query(trinity_query, {'gid1': gid1, 'gid2': gid2})
    return result

def ref_link_SLOOS(n4j, gid1, gid2):
    trinity_query = """
        // Match nodes from Graph A
        MATCH (a)
        WHERE a.gid = $gid1 AND NOT a:Summary
        WITH collect(a) AS GraphA

        // Match nodes from Graph B
        MATCH (b)
        WHERE b.gid = $gid2 AND NOT b:Summary
        WITH GraphA, collect(b) AS GraphB

        // Unwind the nodes to compare each against each
        UNWIND GraphA AS n
        UNWIND GraphB AS m

        // Set the threshold for cosine similarity
        WITH n, m, 0.4 AS threshold

        // Compute cosine similarity and apply the threshold
        WHERE apoc.coll.sort(labels(n)) = apoc.coll.sort(labels(m)) AND n <> m
        WITH n, m, threshold,
            gds.similarity.cosine(n.embedding, m.embedding) AS similarity
        WHERE similarity > threshold

        // Create a relationship based on the condition
        MERGE (m)-[:SLOOS]->(n)

        // Return results
        RETURN n, m
"""

    result = n4j.query(trinity_query, {'gid1': gid1, 'gid2': gid2})
    return result

def ref_link_beigebook(n4j, gid1, gid2):
    trinity_query = """
        // Match nodes from Graph A
        MATCH (a)
        WHERE a.gid = $gid1 AND NOT a:Summary
        WITH collect(a) AS GraphA

        // Match nodes from Graph B
        MATCH (b)
        WHERE b.gid = $gid2 AND NOT b:Summary
        WITH GraphA, collect(b) AS GraphB

        // Unwind the nodes to compare each against each
        UNWIND GraphA AS n
        UNWIND GraphB AS m

        // Set the threshold for cosine similarity
        WITH n, m, 0.4 AS threshold

        // Compute cosine similarity and apply the threshold
        WHERE apoc.coll.sort(labels(n)) = apoc.coll.sort(labels(m)) AND n <> m
        WITH n, m, threshold,
            gds.similarity.cosine(n.embedding, m.embedding) AS similarity
        WHERE similarity > threshold

        // Create a relationship based on the condition
        MERGE (m)-[:beigebook]->(n)

        // Return results
        RETURN n, m
"""
    result = n4j.query(trinity_query, {'gid1': gid1, 'gid2': gid2})
    return result

def str_uuid():
    # Generate a random UUID
    generated_uuid = uuid.uuid4()

    # Convert UUID to a string
    return str(generated_uuid)


