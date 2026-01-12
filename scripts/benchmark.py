#!/usr/bin/env python3
"""
PyVRP benchmark script - equivalent to ExVRP's `mix benchmark`

Runs the same instances with same parameters for comparison.
"""

import sys
import time
from pathlib import Path

from pyvrp import Model, read
from pyvrp.stop import MaxIterations

# Instance config matching ExVRP's benchmark.ex
# Format: (filename, round_func)
# round_func in PyVRP: "round" (default), "trunc", "dimacs" (x10), "exact" (x1000), "none"
INSTANCES = {
    "ok_small": ("OkSmall.txt", "none"),
    "e_n22_k4": ("E-n22-k4.txt", "dimacs"),
    "rc208": ("RC208.vrp", "dimacs"),
    "pr11a": ("PR11A.vrp", "trunc"),
    "c201": ("C201R0.25.vrp", "dimacs"),
    "x101": ("X-n101-50-k13.vrp", "round"),
    "x115": ("X115-HVRP.vrp", "exact"),
    "pr01": ("PR01.vrp", "none"),
    "small_vrpspd": ("SmallVRPSPD.vrp", "round"),
    "p06": ("p06-2-50.vrp", "dimacs"),
    "gtsp": ("50pr439.gtsp", "round"),
    "pr107": ("pr107.tsp", "dimacs"),
}

DATA_DIR = Path(__file__).parent.parent / "tests" / "data"


def run_benchmark(instances: list[str] | None = None, iterations: int = 100, seed: int = 42):
    """Run benchmarks on specified instances."""
    if instances is None:
        instances = list(INSTANCES.keys())

    results = {}

    print(f"\nCollecting solution quality metrics (seed={seed}, iterations={iterations})...\n")

    for name in instances:
        if name not in INSTANCES:
            print(f"  {name}... skipped (unknown instance)")
            continue

        filename, round_func_name = INSTANCES[name]
        path = DATA_DIR / filename

        if not path.exists():
            print(f"  {name}... skipped (file not found: {path})")
            continue

        print(f"  {name}...", end="", flush=True)

        try:
            # Read instance - PyVRP's read() accepts string names for round_func
            data = read(str(path), round_func=round_func_name)
            m = Model.from_data(data)

            start = time.time()
            result = m.solve(stop=MaxIterations(iterations), seed=seed, display=False)
            elapsed = time.time() - start

            sol = result.best
            distance = sol.distance()
            feasible = result.is_feasible()
            num_routes = sol.num_routes()
            num_iters = result.num_iterations

            print(f" done (distance: {distance})")

            results[name] = {
                "distance": distance,
                "feasible": feasible,
                "routes": num_routes,
                "iterations": num_iters,
                "runtime_ms": int(elapsed * 1000),
            }
        except Exception as e:
            print(f" error: {e}")
            results[name] = {"error": str(e)}

    # Print report
    print_report(results, instances)

    return results


def print_report(results: dict, instances: list[str]):
    """Print solution quality report matching ExVRP format."""
    print("")
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║                   PyVRP Solution Quality Report                      ║")
    print("╠══════════════╦══════════════╦══════════╦════════════╦════════════════╣")
    print("║ Instance     ║ Distance     ║ Feasible ║ Routes     ║ Iterations     ║")
    print("╠══════════════╬══════════════╬══════════╬════════════╬════════════════╣")

    for name in instances:
        if name not in results or "error" in results[name]:
            continue
        r = results[name]
        feas = "True" if r["feasible"] else "False"
        print(f"║ {name:<12} ║ {r['distance']:>12} ║ {feas:<8} ║ {r['routes']:>10} ║ {r['iterations']:>14} ║")

    print("╚══════════════╩══════════════╩══════════╩════════════╩════════════════╝")
    print("")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PyVRP benchmark (comparable to ExVRP's mix benchmark)")
    parser.add_argument("--set", action="append", dest="sets", help="Instance to run (can specify multiple)")
    parser.add_argument("--iterations", type=int, default=100, help="Number of iterations (default: 100)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--all", action="store_true", help="Run all instances")
    parser.add_argument("--quick", action="store_true", help="Quick subset (ok_small, e_n22_k4)")

    args = parser.parse_args()

    if args.quick:
        instances = ["ok_small", "e_n22_k4"]
    elif args.sets:
        instances = args.sets
    else:
        instances = None  # all

    run_benchmark(instances=instances, iterations=args.iterations, seed=args.seed)
