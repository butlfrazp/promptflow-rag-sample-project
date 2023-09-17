from .evaluators.rouge import Rouge
from promptflow._internal import tool


@tool
def eval(prediction: str, expected_value: str):
    evaluator = Rouge()
    results = evaluator.compute(predictions=[prediction], references=[expected_value])

    return results


if __name__ == "__main__":
    predictions = "Hello World!"
    expected_values = "Hello World! This is another test"

    results = eval(predictions, expected_values)
    print("running eval..")
    print(results)
