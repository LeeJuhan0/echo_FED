"""
Neo4j context-retrieval functions for FOMC GraphRAG inference.

Each function queries the knowledge graph for a specific document type
(theory papers, FSR, SLOOS, BeigeBook, or the statement itself) and
returns a list of formatted context strings for LLM prompt construction.
"""


def ret_context(n4j, gid: str) -> list[str]:
    """Return intra-graph triples for the FOMC statement identified by *gid*."""
    query = """
        MATCH (n)
        WHERE n.gid = $gid AND NOT n:Summary
        WITH collect(n) AS nodes
        UNWIND nodes AS n
        UNWIND nodes AS m
        MATCH (n)-[r]-(m)
        WHERE n.gid = m.gid AND id(n) < id(m) AND NOT n:Summary AND NOT m:Summary
        WITH n, m, TYPE(r) AS relType
        RETURN n.id AS NodeId1, relType, m.id AS NodeId2
    """
    cont = []
    for r in n4j.query(query, {"gid": gid}):
        cont.append(r["NodeId1"] + r["relType"] + r["NodeId2"])
    return cont


def link_context(n4j, gid: str, i: int, sim_score_median: float) -> list[str]:
    """Return theory-paper references linked to the FOMC statement *gid*."""
    query = """
        MATCH (n)
        WHERE n.gid = $gid AND NOT n:Summary
        MATCH (n)-[r:paragraph]->(m)
        WHERE NOT m:Summary
        MATCH (m)-[s]-(o)
        WHERE NOT o:Summary AND TYPE(s) <> 'paragraph' AND o.sim_score[$i] > $sim_score_median
        RETURN n.id AS NodeId1,
               m.id AS Mid,
               TYPE(r) AS paragraphType,
               collect(DISTINCT {RelationType: type(s), Oid: o.id}) AS Connections
    """
    cont = []
    for r in n4j.query(query, {"gid": gid, "i": i, "sim_score_median": sim_score_median}):
        for ind, connection in enumerate(r["Connections"]):
            cont.append(
                f"Reference {ind}: {r['NodeId1']} has the reference that "
                f"{r['Mid']} {connection['RelationType']} {connection['Oid']}"
            )
    return cont


def link_context_FSR(n4j, gid: str) -> list[str]:
    """Return FSR references linked to the FOMC statement *gid*."""
    query = """
        MATCH (n)
        WHERE n.gid = $gid AND NOT n:Summary
        MATCH (n)-[r:FSR]->(m)
        WHERE NOT m:Summary
        MATCH (m)-[s]-(o)
        WHERE NOT o:Summary AND TYPE(s) <> 'FSR'
        RETURN n.id AS NodeId1,
               m.id AS Mid,
               TYPE(r) AS FSRType,
               collect(DISTINCT {RelationType: type(s), Oid: o.id}) AS Connections
    """
    cont = []
    for r in n4j.query(query, {"gid": gid}):
        for ind, connection in enumerate(r["Connections"]):
            cont.append(
                f"Reference {ind}: {r['NodeId1']} has the reference that "
                f"{r['Mid']} {connection['RelationType']} {connection['Oid']}"
            )
    return cont


def link_context_SLOOS(n4j, gid: str) -> list[str]:
    """Return SLOOS references linked to the FOMC statement *gid* (2-hop)."""
    query = """
        MATCH (n)
        WHERE n.gid = $gid AND NOT n:Summary
        MATCH (n)-[r:SLOOS]->(m)
        WHERE NOT m:Summary
        WITH n, collect({node: m, rel: r}) AS m_list
        UNWIND range(0, size(m_list) - 1) AS i
        WITH n, m_list[i].node AS m, m_list[i].rel AS r, i
        MATCH p = (m)-[*1..2]-(o)
        WHERE NONE(node IN nodes(p) WHERE node:Summary)
          AND NONE(rel IN relationships(p) WHERE type(rel) = 'SLOOS')
        RETURN n.id AS NodeId1,
               m.id AS Mid,
               TYPE(r) AS SLOOSType,
               CASE WHEN i < 3 THEN m.source_text ELSE null END AS SourceText,
               collect(DISTINCT {
                   Hops: length(p),
                   TargetNode: o.id,
                   PathNodes: [node IN nodes(p) | node.id],
                   PathRels:  [rel  IN relationships(p) | type(rel)]
               }) AS Connections
    """
    cont = []
    source_texts = []

    for r in n4j.query(query, {"gid": gid}):
        if r.get("SourceText"):
            source_texts.append(f"Source Text for {r['Mid']}:\n{r['SourceText']}")

        for ind, connection in enumerate(r["Connections"]):
            rels = connection.get("PathRels", [])
            nodes = connection.get("PathNodes", [])
            if len(rels) == 1:
                rel_str = f" -[{rels[0]}]- "
            elif len(rels) > 1:
                rel_str = f" -[{rels[0]}]- {nodes[1]} -[{rels[1]}]- "
            else:
                rel_str = " connects to "
            cont.append(
                f"Reference {ind}: {r['NodeId1']} has the reference that "
                f"{r['Mid']}{rel_str}{nodes[-1]}"
            )

    if source_texts:
        cont.append("\n--- Source Texts ---")
        cont.extend(
            text[:2000] + "..." if len(text) > 2000 else text
            for text in source_texts
        )

    return cont


def link_context_beigebook(n4j, gid: str) -> list[str]:
    """Return Beige Book references linked to the FOMC statement *gid* (2-hop).

    Note: the Cypher query uses ``$gid`` so it correctly scopes results to the
    requested statement rather than a hardcoded value.
    """
    query = """
        MATCH (n)
        WHERE n.gid = $gid AND NOT n:Summary
        MATCH (n)-[r:beigebooknew]->(m)
        WHERE NOT m:Summary
        WITH n, collect({node: m, rel: r}) AS m_list
        UNWIND range(0, size(m_list) - 1) AS i
        WITH n, m_list[i].node AS m, m_list[i].rel AS r, i
        MATCH p = (m)-[*1..2]-(o)
        WHERE NONE(node IN nodes(p) WHERE node:Summary)
          AND NONE(rel IN relationships(p) WHERE type(rel) = 'beigebooknew')
        RETURN n.id AS NodeId1,
               m.id AS Mid,
               TYPE(r) AS beigebookType,
               CASE WHEN i < 3 THEN m.source_text ELSE null END AS SourceText,
               collect(DISTINCT {
                   Hops: length(p),
                   TargetNode: o.id,
                   PathNodes: [node IN nodes(p) | node.id],
                   PathRels:  [rel  IN relationships(p) | type(rel)]
               }) AS Connections
    """
    cont = []
    source_texts = []

    for r in n4j.query(query, {"gid": gid}):
        if r.get("SourceText"):
            source_texts.append(f"Source Text for {r['Mid']}:\n{r['SourceText']}")

        for ind, connection in enumerate(r["Connections"]):
            rels = connection.get("PathRels", [])
            nodes = connection.get("PathNodes", [])
            if len(rels) == 1:
                rel_str = f" -[{rels[0]}]- "
            elif len(rels) > 1:
                rel_str = f" -[{rels[0]}]- {nodes[1]} -[{rels[1]}]- "
            else:
                rel_str = " connects to "
            cont.append(
                f"Reference {ind}: {r['NodeId1']} has the reference that "
                f"{r['Mid']}{rel_str}{nodes[-1]}"
            )

    if source_texts:
        cont.append("\n--- Source Texts ---")
        cont.extend(
            text[:2000] + "..." if len(text) > 2000 else text
            for text in source_texts
        )

    return cont
