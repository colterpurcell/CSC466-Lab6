import numpy as np


def compute_mae(deltas):
    return float(np.mean(np.abs(deltas)))


def compute_metrics(actuals, predicted):
    tp = fp = fn = tn = 0
    for actual, pred in zip(actuals, predicted):
        pred_pos = pred >= 5.0
        actual_pos = actual >= 5.0
        if pred_pos and actual_pos:
            tp += 1
        elif pred_pos and not actual_pos:
            fp += 1
        elif not pred_pos and actual_pos:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': accuracy}


def evaluate_pairs(matrix, method_fn, pairs):
    rows = []
    for uid, iid in pairs:
        actual = matrix[uid, iid]
        predicted = method_fn(matrix, uid, iid)
        rows.append((uid, iid, actual, predicted, actual - predicted))
    return rows


def print_output(rows, mae, metrics):
    print('userID, itemID, Actual_Rating, Predicted_Rating, Delta_Rating')
    for uid, iid, actual, predicted, delta in rows:
        print(f'{uid}, {iid}, {actual:.4f}, {predicted:.4f}, {delta:.4f}')
    print(f'MAE, {mae:.4f}')
    print('Confusion Matrix:')
    print(f'TP: {metrics["tp"]}, FP: {metrics["fp"]}')
    print(f'FN: {metrics["fn"]}, TN: {metrics["tn"]}')
    print(f'Precision: {metrics["precision"]:.4f}')
    print(f'Recall: {metrics["recall"]:.4f}')
    print(f'F-1: {metrics["f1"]:.4f}')
    print(f'Accuracy: {metrics["accuracy"]:.4f}')


def run_random(matrix, method_fn, size, repeats):
    valid_rows, valid_cols = np.where(matrix != 99)
    n_valid = len(valid_rows)
    mae_values = []

    for _ in range(repeats):
        chosen = np.random.choice(n_valid, size=size, replace=size > n_valid)
        pairs = [(int(valid_rows[i]), int(valid_cols[i])) for i in chosen]
        rows = evaluate_pairs(matrix, method_fn, pairs)
        actuals = [r[2] for r in rows]
        predicted = [r[3] for r in rows]
        deltas = [r[4] for r in rows]
        mae = compute_mae(deltas)
        print_output(rows, mae, compute_metrics(actuals, predicted))
        mae_values.append(mae)

    print(f'Mean MAE: {float(np.mean(mae_values)):.4f}')
    print(f'Std MAE:  {float(np.std(mae_values)):.4f}')


def run_list(matrix, method_fn, filename):
    pairs = []
    with open(filename) as f:
        for line in f:
            uid, iid = (int(x.strip()) for x in line.split(','))
            if matrix[uid, iid] != 99:
                pairs.append((uid, iid))

    if not pairs:
        print('MAE, N/A')
        return

    rows = evaluate_pairs(matrix, method_fn, pairs)
    actuals = [r[2] for r in rows]
    predicted = [r[3] for r in rows]
    deltas = [r[4] for r in rows]
    print_output(rows, compute_mae(deltas), compute_metrics(actuals, predicted))
