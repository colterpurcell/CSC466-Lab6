import sys
from loader import load_ratings
from cf_methods import mean_utility, weighted_sum, avg_nnn, weighted_nnn
from evaluator import run_list

METHODS = {
    'mean_utility': mean_utility,
    'weighted_sum': weighted_sum,
    'avg_nnn': avg_nnn,
    'weighted_nnn': weighted_nnn,
}

if len(sys.argv) == 1:
    print('Methods:', ', '.join(METHODS))
    sys.exit(0)

if len(sys.argv) != 3:
    print('Usage: python EvaluateCFList.py Method Filename', file=sys.stderr)
    sys.exit(1)

method_name = sys.argv[1]
if method_name not in METHODS:
    print(f"Unknown method '{method_name}'. Choose from: {', '.join(METHODS)}", file=sys.stderr)
    sys.exit(1)

matrix = load_ratings()
run_list(matrix, METHODS[method_name], sys.argv[2])
