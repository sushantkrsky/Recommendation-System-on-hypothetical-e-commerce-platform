from sklearn.metrics import average_precision_score, ndcg_score

def mean_reciprocal_rank(y_true, y_pred):
    # Implement your MRR function here since scikit-learn does not provide it natively
    pass

def evaluate_model(predictions, ground_truth):
    map_score = average_precision_score(ground_truth, predictions)
    ndcg = ndcg_score([ground_truth], [predictions])
    mrr = mean_reciprocal_rank(ground_truth, predictions)
    return {"MAP": map_score, "NDCG": ndcg, "MRR": mrr}

def mean_reciprocal_rank(y_true, y_pred):
    for index, value in enumerate(y_pred):
        if y_true[index] == 1:  # Check if the prediction is correct
            return 1 / (index + 1)
    return 0


if __name__ == "__main__":
    # Example usage
    predictions = [0.8, 0.6, 0.4]
    ground_truth = [1, 0, 1]
    metrics = evaluate_model(predictions, ground_truth)
    print(metrics)
