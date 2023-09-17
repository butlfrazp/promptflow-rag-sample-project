from typing import List
from promptflow import tool, log_metric
from statistics import median


@tool
def aggregate(processed_results: List[dict]):
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    rougeLSum_scores = []
    for result in processed_results:
        rouge1_scores.append(result["rouge1"])
        rouge2_scores.append(result["rouge2"])
        rougeL_scores.append(result["rougeL"])
        rougeLSum_scores.append(result["rougeLsum"])

    # take the median of the scores
    median_rouge1 = median(rouge1_scores)
    median_rouge2 = median(rouge2_scores)
    median_rougeL = median(rougeL_scores)
    median_rougeLSum = median(rougeLSum_scores)

    # log the median scores
    log_metric(key="median_rouge_rouge1", value=median_rouge1)
    log_metric(key="median_rouge_rouge2", value=median_rouge2)
    log_metric(key="median_rouge_rougeL", value=median_rougeL)
    log_metric(key="median_rouge_rougeLSum", value=median_rougeLSum)

    return {
        "median_rouge_rouge1": median_rouge1,
        "median_rouge_rouge2": median_rouge2,
        "median_rouge_rougeL": median_rougeL,
        "median_rouge_rougeLSum": median_rougeLSum
    }


if __name__ == "__main__":
    processed_results = [
        {
            "rouge1": 0.5,
            "rouge2": 0.5,
            "rougeL": 0.5,
            "rougeLsum": 0.5
        },
        {
            "rouge1": 0.6,
            "rouge2": 0.6,
            "rougeL": 0.6,
            "rougeLsum": 0.6
        },
        {
            "rouge1": 0.7,
            "rouge2": 0.7,
            "rougeL": 0.7,
            "rougeLsum": 0.7
        }
    ]

    result = aggregate(processed_results)
    print(result)
