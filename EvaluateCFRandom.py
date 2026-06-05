import sys
from loader import load_ratings
from cf_methods import mean_utility, weighted_sum, avg_nnn, weighted_nnn
from evaluator import run_random

METHODS = {
    'mean_utility': mean_utility,
    'weighted_sum': weighted_sum,
    'avg_nnn': avg_nnn,
    'weighted_nnn': weighted_nnn,
}

if len(sys.argv) == 1:
    print('Methods:', ', '.join(METHODS))
    sys.exit(0)

if len(sys.argv) != 4:
    print('Usage: python EvaluateCFRandom.py Method Size Repeats', file=sys.stderr)
    sys.exit(1)

method_name = sys.argv[1]
if method_name not in METHODS:
    print(f"Unknown method '{method_name}'. Choose from: {', '.join(METHODS)}", file=sys.stderr)
    sys.exit(1)

try:
    size = int(sys.argv[2])
    repeats = int(sys.argv[3])
    if size < 1 or repeats < 1:
        raise ValueError
except ValueError:
    print('Size and Repeats must be positive integers', file=sys.stderr)
    sys.exit(1)

matrix = load_ratings()
run_random(matrix, METHODS[method_name], size, repeats)
