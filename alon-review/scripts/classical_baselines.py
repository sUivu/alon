#!/usr/bin/env python3
"""
Classical (non-neural) baselines for Minimum Vertex Cover — rebuttal experiments.

Baselines per test graph (all with wall-clock timing):
  - GreedyMVC  : networkx min_weighted_vertex_cover (classical 2-approx greedy)
  - Matching2  : maximal matching endpoints (textbook 2-approximation)
  - LPRounding : LP relaxation of MVC via scipy HiGHS, round x_i >= 0.5 (2-approx)
  - NuMVC      : LibMVC NuMVC local search, 10s cutoff (fixed seed)
  - FastVC     : LibMVC FastVC local search, 10s cutoff (fixed seed)
  - FastWVC    : FastWVC local search, 10s cutoff (fixed seed)
  - Gurobi10s  : Gurobi v13 MIP, 10s time limit, anytime incumbent trace (reference)

Protocol: identical datasets and identical test split as the paper
(train_fraction=0.8, split_seed=0, test = last 10% of the seeded randperm).
Local search solvers receive the proven-optimal size (from Gurobi) as an
early-stop bound when optimality is proven; otherwise they run the full cutoff.

Outputs: results/classical_baselines/<dataset>.json  (per-graph + summary)
Run:  python scripts/classical_baselines.py --all
"""

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import networkx as nx
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.loader import construct_dataset, prepare_alon_dataset  # noqa: E402

SOLVER_DIR = PROJECT_ROOT / "classical_solvers"
NUMVC_BIN = SOLVER_DIR / "libmvc_solver"
FASTWVC_BIN = SOLVER_DIR / "fastwvc" / "mwvc"
# Akiba-Iwata / PACE-style exact branch-and-reduce solver (see
# classical_solvers/pace_vc/). Vendored PACE 2019 exact-track entry, GPL-3.0.
AI_BIN = SOLVER_DIR / "ai_vc_solver"
AI_BUILD = SOLVER_DIR / "pace_vc" / "build.sh"

# ----------------------------------------------------------------------------
# Dataset configs — aligned with the paper's main table (see EXPERIMENT_PARAMS.md)
# ----------------------------------------------------------------------------
def _syn(dataset, gen_n, **kw):
    return dict(kind="generated", dataset=dataset, gen_n=gen_n,
                num_graphs=2000, **kw)

DATASET_CONFIGS = [
    # synthetic, three scales
    ("ER_50_100",  _syn("ErdosRenyi", [50, 100], gen_p=[0.15])),
    ("ER_100_200", _syn("ErdosRenyi", [100, 200], gen_p=[0.15])),
    ("ER_400_500", _syn("ErdosRenyi", [400, 500], gen_p=[0.15])),
    ("BA_50_100",  _syn("BarabasiAlbert", [50, 100], gen_m=[4])),
    ("BA_100_200", _syn("BarabasiAlbert", [100, 200], gen_m=[4])),
    ("BA_400_500", _syn("BarabasiAlbert", [400, 500], gen_m=[4])),
    ("WS_50_100",  _syn("WattsStrogatz", [50, 100], gen_k=[4], gen_p=[0.25])),
    ("WS_100_200", _syn("WattsStrogatz", [100, 200], gen_k=[4], gen_p=[0.25])),
    ("WS_400_500", _syn("WattsStrogatz", [400, 500], gen_k=[4], gen_p=[0.25])),
    ("HK_50_100",  _syn("PowerlawCluster", [50, 100], gen_m=[4], gen_p=[0.25])),
    ("HK_100_200", _syn("PowerlawCluster", [100, 200], gen_m=[4], gen_p=[0.25])),
    ("HK_400_500", _syn("PowerlawCluster", [400, 500], gen_m=[4], gen_p=[0.25])),
    # TU-small real-world
    ("MUTAG", dict(kind="TU", dataset="MUTAG")),
    ("ENZYMES", dict(kind="TU", dataset="ENZYMES")),
    ("PROTEINS", dict(kind="TU", dataset="PROTEINS")),
    ("IMDB-BINARY", dict(kind="TU", dataset="IMDB-BINARY")),
    ("COLLAB", dict(kind="TU", dataset="COLLAB")),
    # hard Forced-RB
    ("RB200", dict(kind="generated", dataset="ForcedRB", gen_n=[6, 15],
                   gen_k=[12, 21], num_graphs=4000)),
    ("RB500", dict(kind="generated", dataset="ForcedRB", gen_n=[20, 34],
                   gen_k=[10, 29], num_graphs=4000)),
]

DEFAULTS = dict(problem_type="vertex_cover", data_seed=0, parallel=0,
                infinite=False, positional_encoding=None, pe_dimension=8,
                gen_m=None, gen_k=None, gen_p=None,
                train_fraction=0.8, split_seed=0)


def build_namespace(cfg):
    ns = argparse.Namespace(**DEFAULTS)
    for k, v in cfg.items():
        setattr(ns, k, v)
    return ns


def load_test_graphs(cfg):
    """Construct the dataset exactly like the paper and return test-graph
    (n, edge_list, dataset_idx, global_node_offset) tuples using the same
    random_split protocol."""
    args = build_namespace(cfg)
    dataset = construct_dataset(args)
    dataset, _ = prepare_alon_dataset(dataset)
    n = len(dataset)
    train_size = int(args.train_fraction * n)
    val_size = (n - train_size) // 2
    gen = torch.Generator().manual_seed(args.split_seed)
    perm = torch.randperm(n, generator=gen).tolist()
    test_idx = perm[train_size + val_size:]
    # global node offsets (prepare_alon_dataset assigns node_global_idx
    # in dataset order)
    if getattr(dataset, "_data_list", None) is not None:
        sizes = [int(d.num_nodes) for d in dataset._data_list]
    else:
        sizes = [int(dataset[i].num_nodes) for i in range(len(dataset))]
    offsets = np.cumsum([0] + sizes)
    graphs = []
    for i in test_idx:
        g = dataset[i]
        ei = g.edge_index.numpy()
        # dedupe undirected edges
        edges = set()
        for u, v in ei.T:
            u, v = int(u), int(v)
            if u != v:
                edges.add((min(u, v), max(u, v)))
        graphs.append((int(g.num_nodes), sorted(edges), int(i), int(offsets[i])))
    return graphs, test_idx


# ----------------------------------------------------------------------------
# Solvers
# ----------------------------------------------------------------------------
def nx_graph(n, edges):
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)
    return G


def check_cover(n, edges, cover):
    covered = set(cover)
    return all((u in covered) or (v in covered) for u, v in edges)


def solve_greedy(n, edges):
    t0 = time.perf_counter()
    G = nx_graph(n, edges)
    cover = nx.algorithms.approximation.min_weighted_vertex_cover(G)
    t = time.perf_counter() - t0
    assert check_cover(n, edges, cover)
    return {"size": int(len(cover)), "time_s": t}


def solve_matching(n, edges):
    t0 = time.perf_counter()
    G = nx_graph(n, edges)
    matching = nx.maximal_matching(G)
    cover = set()
    for u, v in matching:
        cover.add(u)
        cover.add(v)
    t = time.perf_counter() - t0
    assert check_cover(n, edges, cover)
    return {"size": int(len(cover)), "time_s": t}


def solve_lp(n, edges):
    from scipy.optimize import linprog
    t0 = time.perf_counter()
    if n == 0:
        return {"size": 0, "time_s": time.perf_counter() - t0}
    E = len(edges)
    rows, cols = [], []
    for e, (u, v) in enumerate(edges):
        rows.append(e); cols.append(u)
        rows.append(e); cols.append(v)
    A = np.zeros((E, n))
    A[rows, cols] = 1.0
    res = linprog(c=np.ones(n), A_ub=-A, b_ub=-np.ones(E),
                  bounds=(0, 1), method="highs")
    x = res.x
    # strict > 1/2 threshold (half-integral LP solutions have ties at 0.5;
    # taking them would put the whole graph in the cover), then greedy
    # repair of any uncovered edge
    cover = set(np.where(x > 0.5 + 1e-9)[0].tolist())
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    for u, v in edges:
        if u not in cover and v not in cover:
            cover.add(u if deg[u] >= deg[v] else v)
    t = time.perf_counter() - t0
    assert check_cover(n, edges, cover)
    return {"size": int(len(cover)), "time_s": t,
            "lp_optimum": float(res.fun) if res.fun is not None else None}


def write_dimacs(n, edges, path, weighted=False):
    with open(path, "w") as f:
        f.write(f"p edge {n} {len(edges)}\n")
        if weighted:
            for i in range(1, n + 1):
                f.write(f"v {i} 1\n")
        for u, v in edges:
            f.write(f"e {u + 1} {v + 1}\n")


def solve_libmvc(n, edges, solver_name, cutoff_s, optimal_size, seed,
                 tmpdir):
    """Run LibMVC NuMVC/FastVC via the seed-aware driver."""
    graph_file = os.path.join(tmpdir, f"{solver_name}_{os.getpid()}.dimacs")
    write_dimacs(n, edges, graph_file)
    cmd = [str(NUMVC_BIN), solver_name, graph_file, str(optimal_size),
           str(cutoff_s), str(seed)]
    t0 = time.perf_counter()
    try:
        out = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=cutoff_s + 10)
        stdout = out.stdout
    except subprocess.TimeoutExpired:
        return {"size": None, "error": "timeout"}
    wall = time.perf_counter() - t0
    trace = []
    for line in stdout.splitlines():
        m = re.match(r"Better MVC found\.\s+Size:\s+(\d+)\s+Time:\s+(\d+)ms", line)
        if m:
            trace.append([int(m.group(2)) / 1000.0, int(m.group(1))])
    m = re.search(r"RESULT best_cover=(\d+) solve_time_ms=(\d+) "
                  r"total_time_ms=(\d+) ok=(\d+)", stdout)
    if not m:
        return {"size": None, "error": "parse_failed", "stdout": stdout[-500:]}
    size = int(m.group(1))
    return {"size": size,
            "time_s": int(m.group(2)) / 1000.0,
            "total_time_s": int(m.group(3)) / 1000.0,
            "wall_s": wall,
            "trace": trace}


def solve_fastwvc(n, edges, cutoff_s, seed, tmpdir):
    graph_file = os.path.join(tmpdir, f"fastwvc_{os.getpid()}.mwvc")
    write_dimacs(n, edges, graph_file, weighted=True)
    cmd = [str(FASTWVC_BIN), graph_file, str(seed), str(cutoff_s), "0"]
    t0 = time.perf_counter()
    try:
        out = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=cutoff_s + 10)
    except subprocess.TimeoutExpired:
        return {"size": None, "error": "timeout"}
    wall = time.perf_counter() - t0
    # output: "<file>, <best_weight>, <best_comp_time>"
    last = [l for l in out.stdout.strip().splitlines() if "," in l]
    if not last:
        return {"size": None, "error": "parse_failed", "stdout": out.stdout[-300:]}
    parts = last[-1].split(",")
    size = int(float(parts[1].strip()))
    comp_time = float(parts[2].strip())
    return {"size": size, "time_s": comp_time, "wall_s": wall}


def write_dimacs_pace(n, edges, path):
    """PACE-style edge list for the vendored branch-and-reduce solver.

    NOTE: this solver's parser expects the edge lines WITHOUT the 'e' prefix
    (bare 'u v' pairs), so we cannot reuse write_dimacs() here.
    """
    with open(path, "w") as f:
        f.write(f"p edge {n} {len(edges)}\n")
        for u, v in edges:
            f.write(f"{u + 1} {v + 1}\n")


def ensure_ai_bin():
    """Return the exact-solver binary, building it from source if missing."""
    if not AI_BIN.exists():
        if AI_BUILD.exists():
            subprocess.run(["bash", str(AI_BUILD)], check=True)
    return AI_BIN


def solve_akiba_iwata(n, edges, cutoff_s, seed, tmpdir):
    """Akiba-Iwata / PACE-style exact branch-and-reduce MVC solver.

    Exact when it finishes within the cutoff (``exact=True``); otherwise the
    search is aborted at the deadline and the best feasible cover found so far
    is returned with ``truncated=True``. The returned cover is independently
    re-validated in Python (``feasible``) rather than trusted blindly.
    ``seed`` is accepted for CLI symmetry but the solver is deterministic.
    """
    del seed  # solver is deterministic; kept for a uniform solver signature
    if n == 0 or len(edges) == 0:
        return {"size": 0, "time_s": 0.0, "wall_s": 0.0, "exact": True,
                "truncated": False, "feasible": True, "nodes": 0}
    bin_path = ensure_ai_bin()
    graph_file = os.path.join(tmpdir, f"ai_{os.getpid()}.dimacs")
    write_dimacs_pace(n, edges, graph_file)
    t0 = time.perf_counter()
    with open(graph_file) as fin:
        try:
            out = subprocess.run([str(bin_path), str(cutoff_s)],
                                 stdin=fin, capture_output=True, text=True,
                                 timeout=cutoff_s + 30)
        except subprocess.TimeoutExpired:
            return {"size": None, "error": "timeout"}
    wall = time.perf_counter() - t0
    lines = out.stdout.strip().splitlines()
    if not lines:
        return {"size": None, "error": "parse_failed", "stderr": out.stderr[-300:]}
    try:
        fields = dict(kv.split("=") for kv in lines[0].split()[1:])
        size = int(fields["best"])
        time_ms = int(fields["time_ms"])
        nodes = int(fields.get("nodes", 0))
        exact = fields.get("exact") == "1"
        solver_feasible = fields.get("feasible") == "1"
        cover = [int(x) - 1 for x in lines[1:]]
    except (ValueError, KeyError, IndexError) as e:
        return {"size": None, "error": f"parse_failed: {e}",
                "stdout": out.stdout[-300:]}
    feasible = (len(cover) == len(set(cover)) and check_cover(n, edges, cover))
    return {"size": size, "time_s": time_ms / 1000.0, "wall_s": wall,
            "exact": bool(exact), "truncated": bool(not exact),
            "feasible": bool(feasible), "solver_feasible": bool(solver_feasible),
            "nodes": nodes, "solver": "pace_vc"}


def solve_mip_highs(n, edges, time_limit, trace_limits=None):
    """Exact MIP reference via HiGHS (scipy.optimize.milp) — no size limits,
    used as fallback when the Gurobi license rejects large models.
    With trace_limits (list of per-run time limits) builds a coarse anytime
    incumbent trace; stops early once optimality is proven."""
    from scipy.optimize import milp, LinearConstraint, Bounds
    E = len(edges)
    A = np.zeros((E, n))
    for e, (u, v) in enumerate(edges):
        A[e, u] = 1.0
        A[e, v] = 1.0
    c = np.ones(n)
    integrality = np.ones(n)
    bounds = Bounds(0, 1)
    cons = LinearConstraint(A, lb=np.ones(E), ub=np.full(E, np.inf))

    def run(lim):
        t0 = time.perf_counter()
        res = milp(c=c, integrality=integrality, bounds=bounds,
                   constraints=cons,
                   options={"time_limit": lim, "mip_rel_gap": 0.0,
                            "presolve": True, "disp": False})
        return res, time.perf_counter() - t0

    trace = []
    limits = trace_limits if trace_limits else [time_limit]
    last = None
    walls = []
    for lim in limits:
        res, wall = run(lim)
        walls.append(wall)
        last = res
        if res.x is None:  # no incumbent yet
            continue
        obj = int(round(res.fun))
        trace.append([wall, obj])
        if res.status == 0:  # proven optimal
            break
    if last is None or last.x is None:
        return {"size": None, "error": "highs_no_solution"}
    return {"size": int(round(last.fun)), "time_s": sum(walls),
            "optimal_proven": bool(last.status == 0), "trace": trace,
            "solver": "highs", "status": int(last.status)}


def solve_gurobi(n, edges, cutoff_s, trace_limits=None):
    import gurobipy as gp
    from gurobipy import GRB
    t0 = time.perf_counter()
    m = gp.Model("mvc")
    m.setParam("OutputFlag", 0)
    m.setParam("Threads", 2)
    m.setParam("TimeLimit", cutoff_s)
    x = m.addVars(n, vtype=GRB.BINARY, name="x")
    for u, v in edges:
        m.addConstr(x[u] + x[v] >= 1)
    m.setObjective(gp.quicksum(x.values()), GRB.MINIMIZE)
    trace = []
    last_sample = [0.0]

    def cb(model, where):
        if where == GRB.Callback.MIPSOL:
            t = model.cbGet(GRB.Callback.RUNTIME)
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            bnd = model.cbGet(GRB.Callback.MIPSOL_OBJBND)
            trace.append([t, obj, bnd])
            last_sample[0] = t
        elif where == GRB.Callback.MIP:
            t = model.cbGet(GRB.Callback.RUNTIME)
            if t - last_sample[0] >= 0.25:  # sample bound at ~4 Hz
                obj = model.cbGet(GRB.Callback.MIP_OBJBST)
                bnd = model.cbGet(GRB.Callback.MIP_OBJBND)
                trace.append([t, obj, bnd])
                last_sample[0] = t

    try:
        m.optimize(cb)
    except gp.GurobiError as e:
        if "size-limited" in str(e) or "too large" in str(e):
            # license size limit: skip Gurobi for this graph; the reference
            # value for these datasets comes from the paper's Optimal column
            return {"size": None, "error": "license_size_limited"}
        raise
    wall = time.perf_counter() - t0
    if m.SolCount == 0:
        return {"size": None, "error": "no_solution",
                "status": m.Status, "wall_s": wall}
    size = int(round(m.ObjVal))
    optimal_proven = (m.Status == GRB.OPTIMAL)
    if trace:
        trace.append([min(m.Runtime, cutoff_s), int(round(m.ObjVal)),
                      m.ObjBound if hasattr(m, "ObjBound") else None])
    return {"size": size, "time_s": wall, "runtime_s": m.Runtime,
            "optimal_proven": bool(optimal_proven),
            "gap": float(m.MIPGap) if hasattr(m, "MIPGap") else None,
            "trace": trace, "solver": "gurobi"}


def solve_one(args):
    """Solve all baselines for a single graph. args = (n, edges, idx, cutoff_s, seed, tmpdir)"""
    n, edges, idx, cutoff_s, seed, tmpdir = args
    res = {"idx": idx, "n": n, "m": len(edges)}
    try:
        res["greedy"] = solve_greedy(n, edges)
    except Exception as e:
        res["greedy"] = {"error": str(e)}
    try:
        res["matching"] = solve_matching(n, edges)
    except Exception as e:
        res["matching"] = {"error": str(e)}
    try:
        res["lp"] = solve_lp(n, edges)
    except Exception as e:
        res["lp"] = {"error": str(e)}
    try:
        res["gurobi"] = solve_gurobi(n, edges, cutoff_s)
    except Exception as e:
        res["gurobi"] = {"error": str(e)}
    # early-stop bound: pass proven optimal size when available
    opt = (res["gurobi"].get("size")
           if res["gurobi"].get("optimal_proven") else None)
    try:
        res["numvc"] = solve_libmvc(n, edges, "numvc", cutoff_s,
                                    opt if opt else 0, seed, tmpdir)
    except Exception as e:
        res["numvc"] = {"error": str(e)}
    try:
        res["fastvc"] = solve_libmvc(n, edges, "fastvc", cutoff_s,
                                     opt if opt else 0, seed, tmpdir)
    except Exception as e:
        res["fastvc"] = {"error": str(e)}
    try:
        res["fastwvc"] = solve_fastwvc(n, edges, cutoff_s, seed, tmpdir)
    except Exception as e:
        res["fastwvc"] = {"error": str(e)}
    try:
        res["akiba_iwata"] = solve_akiba_iwata(n, edges, cutoff_s, seed, tmpdir)
    except Exception as e:
        res["akiba_iwata"] = {"error": str(e)}
    return res


def summarize(per_graph, reference="gurobi"):
    """Aggregate per-graph solver results into a summary dict.

    Full-subset fields (kept from the original implementation) are computed
    over each solver's OWN subset of graphs:
        count, avg_size, avg_time_s, median_size,
        avg_gap_vs_gurobi10s, n_gap_pairs

    NEW common-subset fields make the table apples-to-apples even when
    coverage differs (the Gurobi license size-limits large graphs, so
    `gurobi` solves far fewer graphs than the local-search baselines):
        coverage                    : "<solved>/<total graphs>"
        n_common                    : size of the common subset
        avg_size_common             : mean size over the common subset
        median_size_common          : median size over the common subset
        avg_time_common             : mean time_s over the common subset
        avg_gap_vs_gurobi10s_common : mean (size - reference size) over the
                                      common subset (reference -> 0.0)
        common_subset_available     : False when the common subset is empty

    COMMON SUBSET DEFINITION
    ------------------------
    The set of graph indices (the dataset `idx`) on which the REFERENCE
    solver (`reference`, default "gurobi") AND every solver present in the
    data returned a non-null `size`. Present solvers are discovered
    dynamically from the per-graph dicts (keys mapping to result dicts with
    a "size"/"error"), so a newly added baseline is picked up automatically
    and no solver list needs to be maintained here. Internal/bookkeeping
    keys ("idx", "n", "m", anything starting with "_") are skipped.

    If the intersection is empty (e.g. ER_400_500 and RB500 where Gurobi
    solved 0 graphs), the common-subset fields are set to null, the
    full-subset fields above remain the fallback, and
    `common_subset_available` is False.
    """
    # ---- discover solver keys dynamically from the data ----------------
    internal = {"idx", "n", "m"}
    solver_keys = set()
    for g in per_graph:
        for k, v in g.items():
            if k in internal or k.startswith("_") or not isinstance(v, dict):
                continue
            if "size" in v or "error" in v:
                solver_keys.add(k)
    solvers = sorted(solver_keys)

    def solved(s):
        """{idx: result_dict} for graphs where solver s returned a size."""
        out = {}
        for g in per_graph:
            r = g.get(s)
            if isinstance(r, dict) and r.get("size") is not None:
                out[g["idx"]] = r
        return out

    solver_solved = {s: solved(s) for s in solvers}

    # reference sizes (for gaps); reference itself must be a present solver
    gurobi_by_idx = {}
    if reference in solver_solved:
        gurobi_by_idx = {i: r["size"]
                         for i, r in solver_solved[reference].items()}

    # common subset = reference-solved AND every present solver solved
    common = set(gurobi_by_idx.keys())
    for s in solvers:
        common &= set(solver_solved[s].keys())
    common_sorted = sorted(common)
    common_available = len(common) > 0

    summary = {}
    for s in solvers:
        rows = list(solver_solved[s].items())
        entry = {"count": len(rows),
                 "coverage": f"{len(rows)}/{len(per_graph)}",
                 "n_common": len(common),
                 "common_subset_available": common_available}
        if not rows:
            entry.update({"avg_size": None, "avg_time_s": None,
                          "median_size": None,
                          "avg_gap_vs_gurobi10s": None, "n_gap_pairs": 0,
                          "avg_size_common": None, "median_size_common": None,
                          "avg_time_common": None,
                          "avg_gap_vs_gurobi10s_common": None})
            summary[s] = entry
            continue
        sizes = [r["size"] for _, r in rows]
        times = [r.get("time_s", 0.0) for _, r in rows]
        # per-graph gap matched on dataset idx (intersection w/ reference)
        gaps = [r["size"] - gurobi_by_idx[i] for i, r in rows
                if i in gurobi_by_idx]
        entry.update({
            "avg_size": float(np.mean(sizes)),
            "avg_time_s": float(np.mean(times)),
            "median_size": float(np.median(sizes)),
            "avg_gap_vs_gurobi10s": (float(np.mean(gaps)) if gaps else None),
            "n_gap_pairs": len(gaps),
        })
        # best size achieved over the solver's solved graphs
        entry["best_size"] = float(np.min(sizes))
        if common_available:
            csizes = [solver_solved[s][i]["size"] for i in common_sorted]
            ctimes = [solver_solved[s][i].get("time_s", 0.0)
                      for i in common_sorted]
            cgaps = [solver_solved[s][i]["size"] - gurobi_by_idx[i]
                     for i in common_sorted if i in gurobi_by_idx]
            entry["avg_size_common"] = float(np.mean(csizes))
            entry["median_size_common"] = float(np.median(csizes))
            entry["avg_time_common"] = float(np.mean(ctimes))
            entry["avg_gap_vs_gurobi10s_common"] = (
                float(np.mean(cgaps)) if cgaps else None)
        else:
            entry["avg_size_common"] = None
            entry["median_size_common"] = None
            entry["avg_time_common"] = None
            entry["avg_gap_vs_gurobi10s_common"] = None
        summary[s] = entry

    summary["_meta"] = {"n_graphs": len(per_graph),
                        "gurobi_solved": len(gurobi_by_idx),
                        "reference_solver": reference,
                        "n_common": len(common),
                        "common_subset_available": common_available}
    return summary


def run_dataset(name, cfg, cutoff_s, seed, workers, out_dir, limit):
    print(f"\n=== {name} ===")
    t0 = time.time()
    graphs, test_idx = load_test_graphs(cfg)
    print(f"dataset graphs={len(graphs)} test_graphs={len(test_idx)} "
          f"(loaded in {time.time()-t0:.1f}s)")
    if limit:
        graphs = graphs[:limit]
    tasks = [(n, edges, idx, cutoff_s, seed, str(out_dir))
             for (n, edges, idx, _off) in graphs]
    per_graph = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
        for i, res in enumerate(ex.map(solve_one, tasks, chunksize=1)):
            per_graph.append(res)
            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{len(tasks)} graphs done "
                      f"({time.time()-t0:.0f}s elapsed)")
    per_graph.sort(key=lambda g: g["idx"])
    summary = summarize(per_graph)
    out_path = out_dir / f"{name}.json"
    json.dump({"dataset": name, "config": cfg, "test_indices": test_idx[:len(per_graph)],
               "cutoff_s": cutoff_s, "seed": seed,
               "per_graph": per_graph, "summary": summary},
              open(out_path, "w"), indent=1)
    print(f"  saved -> {out_path}")
    for s, v in summary.items():
        if v.get("count"):
            print(f"  {s:9s} avg_size={v['avg_size']:8.3f} "
                  f"avg_time={v['avg_time_s']:8.3f}s "
                  f"gap={v.get('avg_gap_vs_gurobi10s')}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=None,
                    help="dataset keys to run (default: all)")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--cutoff", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0,
                    help="limit to first K test graphs (smoke test)")
    ap.add_argument("--out_dir", type=str,
                    default=str(PROJECT_ROOT / "results" / "classical_baselines"))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    names = args.datasets or [k for k, _ in DATASET_CONFIGS]
    configs = dict(DATASET_CONFIGS)
    for name in names:
        if name not in configs:
            print(f"WARNING: unknown dataset {name}, skipped")
            continue
        run_dataset(name, configs[name], args.cutoff, args.seed,
                    args.workers, out_dir, args.limit)


if __name__ == "__main__":
    main()
