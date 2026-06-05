import numpy as np


def clamp(val):
    return float(np.clip(val, -10.0, 10.0))


def item_mean(matrix, item_id):
    item_ratings = matrix[:, item_id]
    rated = item_ratings[item_ratings != 99]
    return float(np.mean(rated)) if len(rated) > 0 else 0.0


def pearson(matrix, user_id, other_user):
    # only items both users have rated count toward the correlation
    both_rated = np.where((matrix[user_id] != 99) & (matrix[other_user] != 99))[0]
    if len(both_rated) < 2:
        return 0.0
    user_ratings = matrix[user_id, both_rated] - matrix[user_id, both_rated].mean()
    other_ratings = matrix[other_user, both_rated] - matrix[other_user, both_rated].mean()
    user_std = user_ratings.std()
    other_std = other_ratings.std()
    if user_std == 0 or other_std == 0:
        return 0.0
    return float(np.dot(user_ratings, other_ratings) / (len(both_rated) * user_std * other_std))


def raters(matrix, user_id, item_id):
    # all users other than user_id who have rated item_id
    return np.where((np.arange(matrix.shape[0]) != user_id) & (matrix[:, item_id] != 99))[0]


def top_n_neighbors(similarities, N):
    # return indices of top-N neighbors by similarity, excluding zeros
    nonzero = np.where(similarities != 0.0)[0]
    if len(nonzero) == 0:
        return nonzero
    return nonzero[np.argsort(np.abs(similarities[nonzero]))[::-1]][:N]


def mean_utility(matrix, user_id, item_id):
    known_rating = matrix[user_id, item_id]
    matrix[user_id, item_id] = 99
    prediction = item_mean(matrix, item_id)
    matrix[user_id, item_id] = known_rating
    return clamp(prediction)


def weighted_sum(matrix, user_id, item_id):
    known_rating = matrix[user_id, item_id]
    matrix[user_id, item_id] = 99

    # weighted sum over all raters
    neighbor_ids = raters(matrix, user_id, item_id)
    numerator = denominator = 0.0
    for neighbor in neighbor_ids:
        similarity = pearson(matrix, user_id, neighbor)
        numerator += similarity * matrix[neighbor, item_id]
        denominator += abs(similarity)

    fallback = item_mean(matrix, item_id)
    matrix[user_id, item_id] = known_rating

    if denominator == 0:
        return clamp(fallback)
    return clamp(numerator / denominator)


def avg_nnn(matrix, user_id, item_id, N=50):
    known_rating = matrix[user_id, item_id]
    matrix[user_id, item_id] = 99

    # find top-N most similar raters, then take the unweighted average of their ratings
    neighbor_ids = raters(matrix, user_id, item_id)
    similarities = np.array([pearson(matrix, user_id, neighbor) for neighbor in neighbor_ids])
    top = top_n_neighbors(similarities, N)

    fallback = item_mean(matrix, item_id)
    matrix[user_id, item_id] = known_rating

    if len(top) == 0:
        return clamp(fallback)
    return clamp(float(np.mean(matrix[neighbor_ids[top], item_id])))


def weighted_nnn(matrix, user_id, item_id, N=50):
    known_rating = matrix[user_id, item_id]
    matrix[user_id, item_id] = 99

    # same as weighted_sum but restricted to top-N neighbors
    neighbor_ids = raters(matrix, user_id, item_id)
    similarities = np.array([pearson(matrix, user_id, neighbor) for neighbor in neighbor_ids])
    top = top_n_neighbors(similarities, N)

    fallback = item_mean(matrix, item_id)
    matrix[user_id, item_id] = known_rating

    if len(top) == 0:
        return clamp(fallback)
    top_similarities = similarities[top]
    top_ratings = matrix[neighbor_ids[top], item_id]
    normalization = np.sum(np.abs(top_similarities))
    if normalization == 0:
        return clamp(fallback)
    return clamp(float(np.sum(top_similarities * top_ratings) / normalization))
