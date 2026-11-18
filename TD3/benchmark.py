"""TD3 - Exercise 4: Comparing Search Algorithms.

Benchmarks Minimax, Alpha-Beta and MCTS on Connect Four:
    1. benchmark(board, algorithm, **kwargs)      -> one run, one result dict
    2. Minimax vs Alpha-Beta for depths 1..5      (same value, fewer nodes)
    3. MCTS with 100, 500, 1000, 2000 simulations (fixed seed)
    4. Automated experiments + summary table
    5. matplotlib plots (nodes vs depth, time vs simulations)

Usage:
    python benchmark.py
"""

import time

from algorithms import alphabeta, mcts, minimax
from game import create_board

DEPTHS = [1, 2, 3, 4, 5]
SIMULATIONS = [100, 500, 1000, 2000]
SEED = 42

ALGORITHMS = {
    "minimax": minimax,
    "alphabeta": alphabeta,
    "mcts": mcts,
}

DISPLAY_NAMES = {
    "minimax": "Minimax",
    "alphabeta": "Alpha-Beta",
    "mcts": "MCTS",
}


# ---------------------------------------------------------------------------
# 1. Benchmarking function
# ---------------------------------------------------------------------------
def benchmark(board, algorithm, **kwargs):
    """Run ``algorithm`` on ``board`` and return its performance metrics.

    ``algorithm`` may be a function from algorithms.py or its name
    ("minimax", "alphabeta", "mcts"). Extra keyword arguments are forwarded
    (e.g. ``depth=4`` or ``num_simulations=500, seed=42``).
    """
    func = ALGORITHMS[algorithm] if isinstance(algorithm, str) else algorithm
    name = DISPLAY_NAMES.get(func.__name__, func.__name__)

    start = time.perf_counter()
    result = func(board, **kwargs)
    elapsed = time.perf_counter() - start

    return {
        "algorithm": name,
        "move": result["move"],
        "score": result["score"],
        "nodes_explored": result["nodes_explored"],
        "time": result.get("time", elapsed),
    }


# ---------------------------------------------------------------------------
# 2. Minimax vs Alpha-Beta
# ---------------------------------------------------------------------------
def compare_minimax_alphabeta(board, depths=DEPTHS):
    """Run Minimax and Alpha-Beta at each depth and check consistency."""
    rows = []
    print("\n=== Minimax vs Alpha-Beta ===")
    for depth in depths:
        mm = benchmark(board, minimax, depth=depth)
        ab = benchmark(board, alphabeta, depth=depth)
        mm["config"] = ab["config"] = f"Depth {depth}"
        mm["param"] = ab["param"] = depth
        rows += [mm, ab]

        same_value = mm["score"] == ab["score"]
        fewer_nodes = ab["nodes_explored"] <= mm["nodes_explored"]
        print(
            f"d={depth}: Minimax move={mm['move'] + 1} score={mm['score']} "
            f"nodes={mm['nodes_explored']} t={mm['time']:.4f}s | "
            f"Alpha-Beta move={ab['move'] + 1} score={ab['score']} "
            f"nodes={ab['nodes_explored']} t={ab['time']:.4f}s | "
            f"same value: {'OK' if same_value else 'FAIL'}, "
            f"AB <= MM nodes: {'OK' if fewer_nodes else 'FAIL'}"
        )
    return rows


# ---------------------------------------------------------------------------
# 3. MCTS simulation budgets
# ---------------------------------------------------------------------------
def compare_mcts(board, budgets=SIMULATIONS, seed=SEED):
    """Run MCTS with several simulation budgets and a fixed seed."""
    rows = []
    print("\n=== MCTS ===")
    for n in budgets:
        res = benchmark(board, mcts, num_simulations=n, seed=seed)
        res["config"] = f"{n} simulations"
        res["param"] = n
        rows.append(res)
        print(
            f"N={n}: move={res['move'] + 1} reward={res['score']} "
            f"nodes={res['nodes_explored']} t={res['time']:.4f}s"
        )
    return rows


# ---------------------------------------------------------------------------
# 4. Automated experiments + table
# ---------------------------------------------------------------------------
def run_all(board=None):
    """Execute every configuration and return the list of results."""
    board = create_board() if board is None else board
    return compare_minimax_alphabeta(board) + compare_mcts(board)


def print_table(results):
    """Print the results as an aligned table.

    Moves are shown 1-7 to match the web visualization.
    """
    headers = ["Algorithm", "Configuration", "Move", "Score", "Nodes", "Time (s)"]
    order = {"Minimax": 0, "Alpha-Beta": 1, "MCTS": 2}
    rows = [
        [
            r["algorithm"],
            r["config"],
            str(r["move"] + 1) if r["move"] is not None else "-",
            str(r["score"]),
            str(r["nodes_explored"]),
            f"{r['time']:.4f}",
        ]
        for r in sorted(results, key=lambda r: (order[r["algorithm"]], r["param"]))
    ]
    widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(headers)]

    def fmt(cells):
        return "  ".join(c.ljust(w) for c, w in zip(cells, widths))

    print("\n" + fmt(headers))
    print("-" * (sum(widths) + 2 * (len(widths) - 1)))
    for row in rows:
        print(fmt(row))
    print("\nNote: Minimax/Alpha-Beta scores are heuristic values, MCTS scores are "
          "estimated rewards; they are not directly comparable.")


# ---------------------------------------------------------------------------
# 5. Plots
# ---------------------------------------------------------------------------
def plot_results(results, filename="benchmark.png"):
    """Plot nodes vs depth (Minimax / Alpha-Beta) and time vs simulations (MCTS)."""
    import matplotlib.pyplot as plt

    def series(name):
        pts = sorted((r["param"], r) for r in results if r["algorithm"] == name)
        return [p for p, _ in pts], [r for _, r in pts]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for name, marker in (("Minimax", "o"), ("Alpha-Beta", "s")):
        x, rs = series(name)
        ax1.plot(x, [r["nodes_explored"] for r in rs], marker=marker, label=name)
    ax1.set_yscale("log")
    ax1.set_xticks(DEPTHS)
    ax1.set_xlabel("Search depth")
    ax1.set_ylabel("Explored nodes (log scale)")
    ax1.set_title("Explored nodes vs search depth")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend()

    x, rs = series("MCTS")
    ax2.plot(x, [r["time"] for r in rs], marker="^", color="tab:green", label="MCTS")
    ax2.set_xlabel("Number of simulations")
    ax2.set_ylabel("Execution time (s)")
    ax2.set_title("MCTS execution time vs simulations")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    print(f"\nPlots saved to {filename}")
    plt.show()


if __name__ == "__main__":
    all_results = run_all()
    print_table(all_results)
    plot_results(all_results)
