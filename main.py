import numpy as np
import numpy.random as npr
import networkx as nx
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List, Callable
import time
import math

# Reproducibility
SEED = 42
rng = np.random.default_rng(SEED)

# Plot settings
plt.rcParams["figure.figsize"] = (6, 4)
plt.rcParams["axes.grid"] = True
plt.rcParams["font.size"] = 11

# Numerical safety
EPS = 1e-12


@dataclass
class ExperimentConfig:
    n: int = 30              # number of vertices
    p: float = 0.30          # edge probability for G(n,p)
    noise: float = 0.05      # probability of flipping each edge/non-edge
    beta: float = 2.0        # inverse temperature for fixed-temperature MCMC
    lam: float = 0.0         # strength of degree-similarity unary term
    
    num_iters: int = 20_000
    burn_in: int = 5_000
    sample_every: int = 10
    
    seed: int = 42


cfg = ExperimentConfig()
print(cfg)


def make_rng(seed: Optional[int] = None) -> np.random.Generator:
    """
    Create a NumPy random number generator.
    """
    if seed is None:
        return np.random.default_rng()
    return np.random.default_rng(seed)


def identity_perm(n: int) -> np.ndarray:
    """
    Return the identity permutation pi[i] = i.
    """
    return np.arange(n, dtype=int)


def random_perm(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Return a random permutation pi of {0, ..., n-1}.
    
    Convention:
        pi[i] is the vertex in graph B matched to vertex i in graph A.
    """
    return rng.permutation(n)


def inverse_perm(pi: np.ndarray) -> np.ndarray:
    """
    Return inverse permutation inv_pi such that inv_pi[pi[i]] = i.
    """
    inv = np.empty_like(pi)
    inv[pi] = np.arange(len(pi))
    return inv


def is_valid_perm(pi: np.ndarray) -> bool:
    """
    Check whether pi is a valid permutation of {0, ..., n-1}.
    """
    n = len(pi)
    return np.array_equal(np.sort(pi), np.arange(n))



# ============================================================
# from here the actual code begins.
# ============================================================

# ============================================================
# Synthetic graph-pair generation
# ============================================================

@dataclass
class GraphPair:
    A: np.ndarray             # first observed graph adjacency matrix
    B: np.ndarray             # second observed graph adjacency matrix, noisy and permuted
    B_clean: np.ndarray       # clean permuted copy of A before noise
    pi_true: np.ndarray       # true permutation: pi_true[i] is the match of i in B
    params: Dict              # parameters used to generate the instance


def sample_er_graph(n: int, p: float, rng: np.random.Generator) -> np.ndarray:
    """
    Sample an undirected Erdős-Rényi graph G(n,p) as a symmetric 0/1 adjacency matrix.

    No self-loops.
    """
    upper = rng.random((n, n)) < p
    upper = np.triu(upper, k=1)
    A = upper + upper.T
    return A.astype(np.int8)


def permute_adjacency(A: np.ndarray, pi: np.ndarray) -> np.ndarray:
    """
    Permute adjacency matrix A according to pi.

    Convention:
        pi[i] = a means vertex i in A is moved to vertex a in the permuted graph.

    Therefore:
        B_clean[pi[i], pi[j]] = A[i, j].
    """
    n = A.shape[0]
    assert A.shape == (n, n)
    assert is_valid_perm(pi)

    B = np.zeros_like(A)
    B[np.ix_(pi, pi)] = A
    return B


def add_resampling_noise(
    B_clean: np.ndarray,
    p: float,
    noise: float,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Add resampling noise to an undirected graph.

    For each unordered pair {i,j}, with probability `noise`,
    resample B[i,j] from Bernoulli(p). Otherwise keep B_clean[i,j].

    This keeps the marginal graph density approximately p.
    """
    n = B_clean.shape[0]
    B = B_clean.copy()

    rows, cols = np.triu_indices(n, k=1)

    resample_mask = rng.random(len(rows)) < noise
    new_edges = (rng.random(len(rows)) < p).astype(np.int8)

    B[rows[resample_mask], cols[resample_mask]] = new_edges[resample_mask]
    B[cols[resample_mask], rows[resample_mask]] = new_edges[resample_mask]

    np.fill_diagonal(B, 0)
    return B.astype(np.int8)


def add_flip_noise(
    B_clean: np.ndarray,
    noise: float,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Add simple edge-flip noise.

    For each unordered pair {i,j}, with probability `noise`,
    replace B[i,j] by 1 - B[i,j].

    This is simple, but for sparse graphs it can make the graph much denser.
    Use resampling noise as the default.
    """
    n = B_clean.shape[0]
    B = B_clean.copy()

    rows, cols = np.triu_indices(n, k=1)
    flip_mask = rng.random(len(rows)) < noise

    B[rows[flip_mask], cols[flip_mask]] = 1 - B[rows[flip_mask], cols[flip_mask]]
    B[cols[flip_mask], rows[flip_mask]] = B[rows[flip_mask], cols[flip_mask]]

    np.fill_diagonal(B, 0)
    return B.astype(np.int8)


def generate_correlated_graph_pair(
    n: int,
    p: float,
    noise: float,
    rng: np.random.Generator,
    noise_model: str = "resample"
) -> GraphPair:
    """
    Generate a pair of correlated graphs (A, B) with a hidden permutation pi_true.

    Steps:
        1. Sample A ~ G(n,p).
        2. Sample a true permutation pi_true.
        3. Form B_clean by permuting A according to pi_true.
        4. Add noise to B_clean to obtain B.

    Args:
        n: number of vertices.
        p: edge probability.
        noise: noise level.
        rng: NumPy random generator.
        noise_model: either "resample" or "flip".

    Returns:
        GraphPair(A, B, B_clean, pi_true, params)
    """
    assert 0 <= p <= 1
    assert 0 <= noise <= 1

    A = sample_er_graph(n, p, rng)
    pi_true = random_perm(n, rng)
    B_clean = permute_adjacency(A, pi_true)

    if noise_model == "resample":
        B = add_resampling_noise(B_clean, p=p, noise=noise, rng=rng)
    elif noise_model == "flip":
        B = add_flip_noise(B_clean, noise=noise, rng=rng)
    else:
        raise ValueError("noise_model must be either 'resample' or 'flip'.")

    params = {
        "n": n,
        "p": p,
        "noise": noise,
        "noise_model": noise_model
    }

    return GraphPair(A=A, B=B, B_clean=B_clean, pi_true=pi_true, params=params)


def graph_density(A: np.ndarray) -> float:
    """
    Return the edge density of an undirected graph adjacency matrix.
    """
    n = A.shape[0]
    return A.sum() / (n * (n - 1))


def check_graph_pair(pair: GraphPair) -> None:
    """
    Print basic sanity checks for a generated graph pair.
    """
    A, B, B_clean, pi_true = pair.A, pair.B, pair.B_clean, pair.pi_true
    n = A.shape[0]

    print("Graph-pair parameters:", pair.params)
    print("A shape:", A.shape)
    print("B shape:", B.shape)
    print("Valid true permutation:", is_valid_perm(pi_true))
    print("A symmetric:", np.array_equal(A, A.T))
    print("B symmetric:", np.array_equal(B, B.T))
    print("A diagonal zero:", np.all(np.diag(A) == 0))
    print("B diagonal zero:", np.all(np.diag(B) == 0))
    print(f"Density of A:       {graph_density(A):.3f}")
    print(f"Density of B_clean: {graph_density(B_clean):.3f}")
    print(f"Density of B:       {graph_density(B):.3f}")
    print("First 10 entries of pi_true:", pi_true[:10])


# test
pair = generate_correlated_graph_pair(
    n=cfg.n,
    p=cfg.p,
    noise=cfg.noise,
    rng=rng,
    noise_model="resample"
)

check_graph_pair(pair)


# ============================================================
# Scoring functions, accuracy, and simple baselines
# ============================================================

@dataclass
class AlignmentResult:
    pi_hat: np.ndarray
    score: float
    accuracy: Optional[float]
    method: str
    extra: Dict


def alignment_accuracy(pi_hat: np.ndarray, pi_true: np.ndarray) -> float:
    """
    Fraction of vertices whose matches are recovered correctly.

    Convention:
        pi_hat[i] and pi_true[i] are both vertices in graph B matched to vertex i in graph A.
    """
    assert len(pi_hat) == len(pi_true)
    assert is_valid_perm(pi_hat)
    assert is_valid_perm(pi_true)
    return np.mean(pi_hat == pi_true)


def edge_overlap_score(A: np.ndarray, B: np.ndarray, pi: np.ndarray) -> float:
    """
    Edge-overlap score:

        S(pi) = sum_{i<j} A[i,j] * B[pi[i], pi[j]]

    This counts how many edges of A are mapped to edges of B.

    This is the default score we use because it avoids the problem that
    non-edges dominate sparse graphs.
    """
    n = A.shape[0]
    assert A.shape == B.shape == (n, n)
    assert is_valid_perm(pi)

    B_aligned = B[np.ix_(pi, pi)]
    return float(np.sum(np.triu(A * B_aligned, k=1)))


def edge_agreement_score(A: np.ndarray, B: np.ndarray, pi: np.ndarray) -> float:
    """
    Edge/non-edge agreement score:

        S(pi) = sum_{i<j} 1{A[i,j] == B[pi[i], pi[j]]}

    This is sometimes natural under a flip-noise likelihood, but for sparse graphs
    it can be misleading because most pairs are non-edges.
    """
    n = A.shape[0]
    assert A.shape == B.shape == (n, n)
    assert is_valid_perm(pi)

    B_aligned = B[np.ix_(pi, pi)]
    upper = np.triu_indices(n, k=1)
    return float(np.sum(A[upper] == B_aligned[upper]))


def centered_edge_score(A: np.ndarray, B: np.ndarray, pi: np.ndarray, p: Optional[float] = None) -> float:
    """
    Centered edge-correlation style score:

        S(pi) = sum_{i<j} (A[i,j] - p) * (B[pi[i], pi[j]] - p)

    This partially corrects for graph density.

    If p is not provided, we use the average density of A and B.
    """
    n = A.shape[0]
    assert A.shape == B.shape == (n, n)
    assert is_valid_perm(pi)

    if p is None:
        p = 0.5 * (graph_density(A) + graph_density(B))

    B_aligned = B[np.ix_(pi, pi)]
    upper = np.triu_indices(n, k=1)

    A_centered = A[upper] - p
    B_centered = B_aligned[upper] - p

    return float(np.sum(A_centered * B_centered))


def degree_node_score(A: np.ndarray, B: np.ndarray, pi: np.ndarray) -> float:
    """
    Degree-similarity unary score:

        S_node(pi) = - sum_i |deg_A(i) - deg_B(pi[i])|

    Higher is better.
    """
    assert A.shape == B.shape
    assert is_valid_perm(pi)

    deg_A = A.sum(axis=1)
    deg_B = B.sum(axis=1)

    return float(-np.sum(np.abs(deg_A - deg_B[pi])))


def total_score(
    A: np.ndarray,
    B: np.ndarray,
    pi: np.ndarray,
    beta_edge_score: str = "overlap",
    lam: float = 0.0
) -> float:
    """
    Total score used by the sampler.

    Default:
        edge overlap + lambda * degree similarity

    Args:
        beta_edge_score:
            "overlap"   -> edge_overlap_score
            "agreement" -> edge_agreement_score
            "centered"  -> centered_edge_score
        lam:
            weight for degree-node score.
    """
    if beta_edge_score == "overlap":
        s_edge = edge_overlap_score(A, B, pi)
    elif beta_edge_score == "agreement":
        s_edge = edge_agreement_score(A, B, pi)
    elif beta_edge_score == "centered":
        s_edge = centered_edge_score(A, B, pi)
    else:
        raise ValueError("beta_edge_score must be 'overlap', 'agreement', or 'centered'.")

    s_node = degree_node_score(A, B, pi)

    return float(s_edge + lam * s_node)


def random_baseline(pair: GraphPair, rng: np.random.Generator) -> AlignmentResult:
    """
    Random permutation baseline.
    """
    pi_hat = random_perm(pair.A.shape[0], rng)
    score = total_score(pair.A, pair.B, pi_hat, lam=0.0)
    acc = alignment_accuracy(pi_hat, pair.pi_true)

    return AlignmentResult(
        pi_hat=pi_hat,
        score=score,
        accuracy=acc,
        method="random",
        extra={}
    )


def degree_hungarian_baseline(pair: GraphPair) -> AlignmentResult:
    """
    Degree/Hungarian baseline.

    We solve:
        min_pi sum_i |deg_A(i) - deg_B(pi[i])|

    using scipy.optimize.linear_sum_assignment.
    """
    A, B = pair.A, pair.B
    n = A.shape[0]

    deg_A = A.sum(axis=1)
    deg_B = B.sum(axis=1)

    # Cost matrix: rows are vertices of A, columns are vertices of B.
    C = np.abs(deg_A[:, None] - deg_B[None, :])

    row_ind, col_ind = linear_sum_assignment(C)

    pi_hat = np.empty(n, dtype=int)
    pi_hat[row_ind] = col_ind

    score = total_score(A, B, pi_hat, lam=0.0)
    acc = alignment_accuracy(pi_hat, pair.pi_true)

    return AlignmentResult(
        pi_hat=pi_hat,
        score=score,
        accuracy=acc,
        method="degree_hungarian",
        extra={"degree_cost": float(C[row_ind, col_ind].sum())}
    )


def true_alignment_result(pair: GraphPair) -> AlignmentResult:
    """
    Score and accuracy of the true hidden permutation.
    This is not a usable algorithm; it is only for sanity checks.
    """
    pi = pair.pi_true.copy()
    score = total_score(pair.A, pair.B, pi, lam=0.0)
    acc = alignment_accuracy(pi, pair.pi_true)

    return AlignmentResult(
        pi_hat=pi,
        score=score,
        accuracy=acc,
        method="true_alignment",
        extra={}
    )


def summarize_alignment_result(result: AlignmentResult) -> None:
    """
    Pretty-print an AlignmentResult.
    """
    print(f"Method:   {result.method}")
    print(f"Score:    {result.score:.3f}")
    if result.accuracy is not None:
        print(f"Accuracy: {result.accuracy:.3f}")
    if result.extra:
        print("Extra:", result.extra)




# ============================================================
# Sanity check: true alignment vs baselines
# ============================================================

pair = generate_correlated_graph_pair(
    n=cfg.n,
    p=cfg.p,
    noise=cfg.noise,
    rng=rng,
    noise_model="resample"
)

check_graph_pair(pair)

print("\n--- True alignment ---")
true_res = true_alignment_result(pair)
summarize_alignment_result(true_res)

print("\n--- Random baseline ---")
rand_res = random_baseline(pair, rng)
summarize_alignment_result(rand_res)

print("\n--- Degree/Hungarian baseline ---")
deg_res = degree_hungarian_baseline(pair)
summarize_alignment_result(deg_res)




# ============================================================
# Compare true score to many random permutations
# ============================================================

def random_score_distribution(
    pair: GraphPair,
    num_random: int,
    rng: np.random.Generator,
    lam: float = 0.0
) -> np.ndarray:
    """
    Score many random permutations.
    """
    n = pair.A.shape[0]
    scores = np.zeros(num_random)

    for t in range(num_random):
        pi = random_perm(n, rng)
        scores[t] = total_score(pair.A, pair.B, pi, lam=lam)

    return scores


random_scores = random_score_distribution(pair, num_random=1000, rng=rng, lam=0.0)
true_score = true_res.score
deg_score = deg_res.score

plt.figure(figsize=(7, 4))
plt.hist(random_scores, bins=30, alpha=0.75, label="Random permutations")
plt.axvline(true_score, linestyle="--", linewidth=2, label="True permutation")
plt.axvline(deg_score, linestyle="--", linewidth=2, label="Degree/Hungarian")
plt.xlabel("Alignment score")
plt.ylabel("Count")
plt.title("Score of true alignment compared to random permutations")
plt.legend()
plt.show()

print(f"Random score mean: {random_scores.mean():.3f}")
print(f"Random score std:  {random_scores.std():.3f}")
print(f"True score:        {true_score:.3f}")
print(f"Degree score:      {deg_score:.3f}")