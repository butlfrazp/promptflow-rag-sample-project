from typing import List
from promptflow import tool, log_metric
from statistics import median


@tool
def aggregate(processed_results: List[dict]):
    precision_scores = []
    recall_scores = []
    f1_scores = []
    for result in processed_results:
        precision_scores.append(result["precision"])
        recall_scores.append(result["recall"])
        f1_scores.append(result["f1"])

    # take the median of the scores
    median_precision = median(precision_scores)
    median_recall = median(recall_scores)
    median_f1 = median(f1_scores)

    # log the median scores
    log_metric(key="median_bert_precision", value=median_precision)
    log_metric(key="median_bert_recall", value=median_recall)
    log_metric(key="median_bert_f1", value=median_f1)

    return {
        "median_bert_precision": median_precision,
        "median_bert_recall": median_recall,
        "median_bert_f1": median_f1
    }


if __name__ == "__main__":
    processed_results = [
        {
            "precision": 0.5,
            "recall": 0.5,
            "f1": 0.5
        },
        {
            "precision": 0.6,
            "recall": 0.6,
            "f1": 0.6
        },
        {
            "precision": 0.7,
            "recall": 0.7,
            "f1": 0.7
        },
        {
            "precision": 0.8,
            "recall": 0.8,
            "f1": 0.8
        }
    ]

    result = aggregate(processed_results)
    print(result)
