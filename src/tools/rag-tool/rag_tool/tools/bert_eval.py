from typing import List
from promptflow._internal import tool
from bert_score import score


@tool
def eval(
    prediction: str,
    expected_value: str,
):
    bert_precision, bert_recall, bert_f1 = score(
        refs = [prediction],
        cands = [expected_value],
        lang = "en"
    )

    return {
        "precision": bert_precision.numpy().mean().item(),
        "recall": bert_recall.numpy().mean().item(),
        "f1": bert_f1.numpy().mean().item()
    }


if __name__ == "__main__":
    predictions = "This is the most invalid answer ever"
    expected_values = "Same"

    results = eval(predictions, expected_values)
    print("running eval..")
    print(results)