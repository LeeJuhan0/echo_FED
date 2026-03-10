"""
Graph utility functions for FOMC GraphRAG.

Handles embedding generation, node property injection, similarity scores,
and cross-graph node linking via Neo4j.
"""

import time
import random
import os

from openai import (
    OpenAI,
    APITimeoutError,
    APIConnectionError,
    InternalServerError,
    RateLimitError,
)

_MAX_ATTEMPTS = 6
_BASE_DELAY = 2.0


def get_embedding(text: str, mod: str = "text-embedding-3-small") -> list[float] | None:
    """Generate an OpenAI embedding for *text* with exponential-backoff retry.

    Returns the embedding vector on success, or ``None`` if all retries fail.
    Raises immediately on non-retryable errors.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    for attempt in range(_MAX_ATTEMPTS):
        try:
            response = client.embeddings.create(input=text, model=mod)
            return response.data[0].embedding
        except (APITimeoutError, APIConnectionError, InternalServerError, RateLimitError) as e:
            delay = _BASE_DELAY * (2 ** attempt) + random.uniform(0.1, 0.5)
            print(f"[Embedding Attempt {attempt + 1}] {str(e)[:60]}... retrying in {delay:.2f}s")
            time.sleep(delay)
        except Exception as e:
            print(f"Non-retryable embedding error: {e}")
            raise

    print("Embedding call exhausted all retries.")
    return None


# ---------------------------------------------------------------------------
# Node / relationship property helpers
# ---------------------------------------------------------------------------

def add_text_property(graph_element, text_value: str, field: str = "source_text"):
    """Attach *text_value* as a property on every node in *graph_element*."""
    for node in graph_element.nodes:
        node.properties[field] = text_value
    return graph_element


def add_sim_score(graph_element, sim_score):
    """Attach *sim_score* to every node and relationship in *graph_element*."""
    for node in graph_element.nodes:
        node.properties["sim_score"] = sim_score
    for rel in graph_element.relationships:
        rel.properties["sim_score"] = sim_score
    return graph_element


def add_gid(graph_element, gid: str):
    """Attach *gid* to every node and relationship in *graph_element*."""
    for node in graph_element.nodes:
        node.properties["gid"] = gid
    for rel in graph_element.relationships:
        rel.properties["gid"] = gid
    return graph_element


def add_ge_emb(graph_element):
    """Compute and attach context-aware embeddings to every node."""
    for node in graph_element.nodes:
        node_id = node.id
        node_type = getattr(node, "type", "Entity")

        spo_list = []
        for rel in graph_element.relationships:
            src_id = (
                rel.subj.id if hasattr(rel, "subj") else getattr(rel.source, "id", str(rel.source))
            )
            tgt_id = (
                rel.obj.id if hasattr(rel, "obj") else getattr(rel.target, "id", str(rel.target))
            )
            rel_type = getattr(rel, "type", "RELATED_TO")
            if src_id == node_id or tgt_id == node_id:
                spo_list.append(f"[{src_id} - {rel_type} -> {tgt_id}]")

        spo_context = ", ".join(spo_list) if spo_list else "No direct relationships."
        contextualized_text = (
            f"Entity: {node_id} ({node_type})\n"
            f"Relationships: {spo_context}"
        )
        emb = get_embedding(contextualized_text)
        node.properties["embedding"] = emb
    return graph_element


# ---------------------------------------------------------------------------
# Neo4j helpers
# ---------------------------------------------------------------------------

def fetch_texts(n4j):
    query = "MATCH (n) RETURN n.id AS id"
    return n4j.query(query)


def add_embeddings(n4j, node_id: str, embedding: list[float]):
    query = "MATCH (n) WHERE n.id = $node_id SET n.embedding = $embedding"
    n4j.query(query, params={"node_id": node_id, "embedding": embedding})


def add_nodes_emb(n4j):
    nodes = fetch_texts(n4j)
    for node in nodes:
        if node["id"]:
            embedding = get_embedding(node["id"])
            add_embeddings(n4j, node["id"], embedding)


def merge_similar_nodes(n4j, gid: str = None):
    if gid:
        merge_query = """
            WITH 0.5 AS threshold
            MATCH (n), (m)
            WHERE NOT n:Summary AND NOT m:Summary AND n.gid = m.gid AND n.gid = $gid AND n<>m
              AND apoc.coll.sort(labels(n)) = apoc.coll.sort(labels(m))
            WITH n, m, gds.similarity.cosine(n.embedding, m.embedding) AS similarity
            WHERE similarity > threshold
            WITH head(collect([n,m])) AS nodes
            CALL apoc.refactor.mergeNodes(nodes, {properties: 'overwrite', mergeRels: true})
            YIELD node
            RETURN count(*)
        """
        return n4j.query(merge_query, {"gid": gid})

    merge_query = """
        WITH 0.5 AS threshold
        MATCH (n), (m)
        WHERE NOT n:Summary AND NOT m:Summary AND n<>m
          AND apoc.coll.sort(labels(n)) = apoc.coll.sort(labels(m))
        WITH n, m, gds.similarity.cosine(n.embedding, m.embedding) AS similarity
        WHERE similarity > threshold
        WITH head(collect([n,m])) AS nodes
        CALL apoc.refactor.mergeNodes(nodes, {properties: 'overwrite', mergeRels: true})
        YIELD node
        RETURN count(*)
    """
    return n4j.query(merge_query)


# ---------------------------------------------------------------------------
# Cross-graph linking
# ---------------------------------------------------------------------------

def ref_link(n4j, gid1: str, gid2: str):
    """Create :paragraph relationships from *gid1* nodes to similar *gid2* nodes."""
    query = """
        MATCH (a) WHERE a.gid = $gid1 AND NOT a:Summary
        WITH collect(a) AS GraphA
        MATCH (b) WHERE b.gid = $gid2 AND NOT b:Summary
        WITH GraphA, collect(b) AS GraphB
        UNWIND GraphA AS n
        UNWIND GraphB AS m
        WITH n, m, 0.5 AS threshold
        WHERE apoc.coll.sort(labels(n)) = apoc.coll.sort(labels(m)) AND n <> m
        WITH n, m, threshold, gds.similarity.cosine(n.embedding, m.embedding) AS similarity
        WHERE similarity > threshold
        MERGE (m)-[:paragraph]->(n)
        RETURN n, m
    """
    return n4j.query(query, {"gid1": gid1, "gid2": gid2})


def ref_link_FSR(n4j, gid1: str, gid2: str):
    """Create :FSR relationships from *gid1* nodes to similar *gid2* nodes."""
    query = """
        MATCH (a) WHERE a.gid = $gid1 AND NOT a:Summary
        WITH collect(a) AS GraphA
        MATCH (b) WHERE b.gid = $gid2 AND NOT b:Summary
        WITH GraphA, collect(b) AS GraphB
        UNWIND GraphA AS n
        UNWIND GraphB AS m
        WITH n, m, 0.5 AS threshold
        WHERE apoc.coll.sort(labels(n)) = apoc.coll.sort(labels(m)) AND n <> m
        WITH n, m, threshold, gds.similarity.cosine(n.embedding, m.embedding) AS similarity
        WHERE similarity > threshold
        MERGE (m)-[:FSR]->(n)
        RETURN n, m
    """
    return n4j.query(query, {"gid1": gid1, "gid2": gid2})


def ref_link_SLOOS(n4j, gid1: str, gid2: str):
    """Create :SLOOS relationships from *gid1* nodes to similar *gid2* nodes."""
    query = """
        MATCH (a) WHERE a.gid = $gid1 AND NOT a:Summary
        WITH collect(a) AS GraphA
        MATCH (b) WHERE b.gid = $gid2 AND NOT b:Summary
        WITH GraphA, collect(b) AS GraphB
        UNWIND GraphA AS n
        UNWIND GraphB AS m
        WITH n, m, 0.4 AS threshold
        WHERE apoc.coll.sort(labels(n)) = apoc.coll.sort(labels(m)) AND n <> m
        WITH n, m, threshold, gds.similarity.cosine(n.embedding, m.embedding) AS similarity
        WHERE similarity > threshold
        MERGE (m)-[:SLOOS]->(n)
        RETURN n, m
    """
    return n4j.query(query, {"gid1": gid1, "gid2": gid2})


def ref_link_beigebook(n4j, gid1: str, gid2: str):
    """Create :beigebook relationships from *gid1* nodes to similar *gid2* nodes."""
    query = """
        MATCH (a) WHERE a.gid = $gid1 AND NOT a:Summary
        WITH collect(a) AS GraphA
        MATCH (b) WHERE b.gid = $gid2 AND NOT b:Summary
        WITH GraphA, collect(b) AS GraphB
        UNWIND GraphA AS n
        UNWIND GraphB AS m
        WITH n, m, 0.4 AS threshold
        WHERE apoc.coll.sort(labels(n)) = apoc.coll.sort(labels(m)) AND n <> m
        WITH n, m, threshold, gds.similarity.cosine(n.embedding, m.embedding) AS similarity
        WHERE similarity > threshold
        MERGE (m)-[:beigebook]->(n)
        RETURN n, m
    """
    return n4j.query(query, {"gid1": gid1, "gid2": gid2})
