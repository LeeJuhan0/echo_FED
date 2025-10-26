from __future__ import annotations
import itertools
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple, Optional

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy import stats
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh


@dataclass
class GraphCompareResult:
    # 겹침/크기
    n_nodes_1: int
    n_nodes_2: int
    n_edges_1: int
    n_edges_2: int
    node_jaccard: Optional[float]
    edge_jaccard_on_common_nodes: Optional[float]
    # 분포 KS
    ks_degree_stat: Optional[float]
    ks_degree_p: Optional[float]
    ks_pagerank_stat: Optional[float]
    ks_pagerank_p: Optional[float]
    # 스펙트럼
    spectral_l2: Optional[float]
    # ASE+MMD
    mmd2_stat: Optional[float]
    mmd_pvalue: Optional[float]


def union_layout(G1: nx.Graph, G2: nx.Graph, k: float = 0.8, seed: int = 42) -> Dict:
    """두 그래프 합집합으로 spring layout 좌표 계산."""
    U = nx.Graph()
    U.add_nodes_from(G1.nodes(data=True))
    U.add_edges_from(G1.edges(data=True))
    U.add_nodes_from(G2.nodes(data=True))
    U.add_edges_from(G2.edges(data=True))
    pos = nx.spring_layout(U, k=k, seed=seed)
    return pos


def draw_small_multiples(G1: nx.Graph, G2: nx.Graph, pos: Dict, out_path: str):
    """같은 좌표로 A/B 스냅샷을 좌/우 배치."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, G, title in [(axes[0], G1, "Graph A"), (axes[1], G2, "Graph B")]:
        ax.axis("off")
        nx.draw_networkx_edges(G, pos, ax=ax, width=0.8, edge_color="#c0c0c0", alpha=0.7)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=50, node_color="#4c78a8", edgecolors="white", linewidths=0.3)
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def draw_diff_overlay(G1: nx.Graph, G2: nx.Graph, pos: Dict, out_path: str):
    """유지(회색), 신규(녹색), 제거(빨강) 엣지/노드로 오버레이."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis("off")

    nodes1, nodes2 = set(G1.nodes()), set(G2.nodes())
    edges1, edges2 = set(G1.edges()), set(G2.edges())

    nodes_keep = list(nodes1 & nodes2)
    nodes_add = list(nodes2 - nodes1)
    nodes_del = list(nodes1 - nodes2)

    edges_keep = list(edges1 & edges2)
    edges_add = list(edges2 - edges1)
    edges_del = list(edges1 - edges2)

    # 유지
    nx.draw_networkx_edges(nx.Graph(list(edges_keep)), pos, ax=ax, width=0.8, edge_color="#bdbdbd", alpha=0.6)
    nx.draw_networkx_nodes(nx.Graph(nodes_keep), pos, ax=ax, node_size=40, node_color="#bdbdbd")

    # 신규(초록)
    if edges_add:
        nx.draw_networkx_edges(nx.Graph(list(edges_add)), pos, ax=ax, width=1.6, edge_color="#2ecc71", alpha=0.9)
    if nodes_add:
        nx.draw_networkx_nodes(nx.Graph(nodes_add), pos, ax=ax, node_size=70, node_color="#2ecc71", edgecolors="#145a32", linewidths=0.8)

    # 제거(빨강, 잔상)
    if edges_del:
        nx.draw_networkx_edges(nx.Graph(list(edges_del)), pos, ax=ax, width=1.2, edge_color="#e74c3c", alpha=0.55, style="dashed")
    if nodes_del:
        nx.draw_networkx_nodes(nx.Graph(nodes_del), pos, ax=ax, node_size=60, node_color="#e74c3c", alpha=0.55)

    ax.set_title("Diff overlay (green=new in B, red=removed from A)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def _common_node_order(G1: nx.Graph, G2: nx.Graph) -> List:
    common = sorted(set(G1.nodes()).intersection(G2.nodes()))
    return common


def _adjacency_in_order(G: nx.Graph, nodes: List) -> csr_matrix:
    # NetworkX의 adjacency_matrix는 노드 순서 지정 가능
    return nx.to_scipy_sparse_array(G, nodelist=nodes, format="csr")


def laplacian_spectral_distance(G1: nx.Graph, G2: nx.Graph, k: int = 50) -> Optional[float]:
    """정규화 라플라시안의 하위 k 고유값 스펙트럼 L2 거리(공통 노드만)."""
    nodes = _common_node_order(G1, G2)
    if len(nodes) < 3:
        return None
    # 정규화 라플라시안
    L1 = nx.normalized_laplacian_matrix(G1, nodelist=nodes)
    L2 = nx.normalized_laplacian_matrix(G2, nodelist=nodes)
    k_use = min(k, len(nodes) - 1)
    vals1 = np.sort(eigsh(L1, k=k_use, which="SM", return_eigenvectors=False))
    vals2 = np.sort(eigsh(L2, k=k_use, which="SM", return_eigenvectors=False))
    return float(np.linalg.norm(vals1 - vals2))


def ase_embedding(G: nx.Graph, nodes: List, d: int = 16) -> np.ndarray:
    """Adjacency Spectral Embedding: 상위 d 고유쌍으로 임베딩."""
    A = _adjacency_in_order(G, nodes).astype(float)
    # 대칭 가정이 약하면 svds 사용 고려; 여기서는 eigsh로 간단화
    # 양/음 고유값의 절댓값 루트로 스케일
    vals, vecs = eigsh(A, k=min(d, A.shape[0]-1), which="LA")
    X = vecs @ np.diag(np.sqrt(np.abs(vals)))
    return X


def _rbf_kernel(X: np.ndarray, Y: np.ndarray, gamma: Optional[float] = None) -> np.ndarray:
    if gamma is None:
        # median heuristic
        Z = np.vstack([X, Y])
        dists = np.linalg.norm(Z[:, None, :] - Z[None, :, :], axis=2)
        med = np.median(dists[dists > 0])
        gamma = 1.0 / (2 * (med**2 + 1e-12))
    XX = np.sum(X**2, axis=1, keepdims=True)
    YY = np.sum(Y**2, axis=1, keepdims=True)
    K = np.exp(-gamma * (XX - 2 * X @ Y.T + YY.T))
    return K


def mmd_unbiased(Kxx: np.ndarray, Kyy: np.ndarray, Kxy: np.ndarray) -> float:
    n = Kxx.shape[0]
    m = Kyy.shape[0]
    term_x = (np.sum(Kxx) - np.trace(Kxx)) / (n * (n - 1))
    term_y = (np.sum(Kyy) - np.trace(Kyy)) / (m * (m - 1))
    term_xy = (2 * np.sum(Kxy)) / (n * m)
    return float(term_x + term_y - term_xy)


def mmd_permutation_test(X: np.ndarray, Y: np.ndarray, B: int = 500, seed: int = 0) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    Kxx = _rbf_kernel(X, X)
    Kyy = _rbf_kernel(Y, Y)
    Kxy = _rbf_kernel(X, Y)
    stat = mmd_unbiased(Kxx, Kyy, Kxy)

    Z = np.vstack([X, Y])
    n, m = len(X), len(Y)
    count = 0
    for _ in range(B):
        idx = rng.permutation(n + m)
        Xb, Yb = Z[idx[:n]], Z[idx[n:]]
        Kxx_b = _rbf_kernel(Xb, Xb)
        Kyy_b = _rbf_kernel(Yb, Yb)
        Kxy_b = _rbf_kernel(Xb, Yb)
        stat_b = mmd_unbiased(Kxx_b, Kyy_b, Kxy_b)
        if stat_b >= stat - 1e-12:
            count += 1
    pval = (count + 1) / (B + 1)
    return stat, pval


def compare_graphs(G1: nx.Graph, G2: nx.Graph, d_embed: int = 16) -> GraphCompareResult:
    # 크기/겹침
    n1, n2 = G1.number_of_nodes(), G2.number_of_nodes()
    e1, e2 = G1.number_of_edges(), G2.number_of_edges()
    nodes1, nodes2 = set(G1.nodes()), set(G2.nodes())
    node_jacc = len(nodes1 & nodes2) / len(nodes1 | nodes2) if (nodes1 | nodes2) else None

    # 공통 노드로 제한한 엣지 Jaccard
    common = nodes1 & nodes2
    if len(common) >= 2:
        E1c = set((u, v) for u, v in G1.edges() if u in common and v in common)
        E2c = set((u, v) for u, v in G2.edges() if u in common and v in common)
        edge_jacc = len(E1c & E2c) / len(E1c | E2c) if (E1c | E2c) else 1.0
    else:
        edge_jacc = None

    # 분포 KS(차수, PageRank)
    if common:
        order = sorted(common)
        deg1 = np.array([G1.degree(n) for n in order])
        deg2 = np.array([G2.degree(n) for n in order])
        ks_d = stats.ks_2samp(deg1, deg2)
        # PageRank(동일 damping, 반복)
        pr1 = nx.pagerank(G1, alpha=0.85, max_iter=100)
        pr2 = nx.pagerank(G2, alpha=0.85, max_iter=100)
        pr1v = np.array([pr1[n] for n in order])
        pr2v = np.array([pr2[n] for n in order])
        ks_pr = stats.ks_2samp(pr1v, pr2v)
    else:
        ks_d = stats.KstestResult(statistic=np.nan, pvalue=np.nan)
        ks_pr = stats.KstestResult(statistic=np.nan, pvalue=np.nan)

    # 스펙트럼 거리
    spec_l2 = laplacian_spectral_distance(G1, G2, k=50)

    # ASE + MMD 2표본 검정
    if len(common) >= 10:
        order = sorted(common)
        X1 = ase_embedding(G1, order, d=d_embed)
        X2 = ase_embedding(G2, order, d=d_embed)
        mmd_stat, mmd_p = mmd_permutation_test(X1, X2, B=500, seed=1)
    else:
        mmd_stat, mmd_p = None, None

    return GraphCompareResult(
        n_nodes_1=n1, n_nodes_2=n2, n_edges_1=e1, n_edges_2=e2,
        node_jaccard=node_jacc,
        edge_jaccard_on_common_nodes=edge_jacc,
        ks_degree_stat=float(ks_d.statistic) if not np.isnan(ks_d.statistic) else None,
        ks_degree_p=float(ks_d.pvalue) if not np.isnan(ks_d.pvalue) else None,
        ks_pagerank_stat=float(ks_pr.statistic) if not np.isnan(ks_pr.statistic) else None,
        ks_pagerank_p=float(ks_pr.pvalue) if not np.isnan(ks_pr.pvalue) else None,
        spectral_l2=spec_l2,
        mmd2_stat=mmd_stat,
        mmd_pvalue=mmd_p,
    )


def demo():
    # 예시: A에서 몇 개 엣지 추가/삭제하여 B 생성
    G1 = nx.erdos_renyi_graph(200, 0.03, seed=0)
    G2 = G1.copy()
    # 엣지 편집
    to_remove = list(itertools.islice(G2.edges(), 150))
    G2.remove_edges_from(to_remove)
    rng = np.random.default_rng(1)
    nodes = list(G2.nodes())
    new_edges = [(int(rng.choice(nodes)), int(rng.choice(nodes))) for _ in range(150)]
    G2.add_edges_from([(u, v) for u, v in new_edges if u != v])

    pos = union_layout(G1, G2, k=0.9, seed=42)
    draw_small_multiples(G1, G2, pos, out_path="graphs_small_multiples.png")
    draw_diff_overlay(G1, G2, pos, out_path="graphs_diff_overlay.png")

    res = compare_graphs(G1, G2, d_embed=16)
    print("=== Graph comparison ===")
    print(res)


if __name__ == "__main__":
    demo()