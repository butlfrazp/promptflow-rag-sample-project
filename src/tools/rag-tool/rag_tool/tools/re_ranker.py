from typing import List
from promptflow._internal import tool
from sentence_transformers import CrossEncoder


@tool
def re_ranker(
    inputs: List[object],
    query: str,
    model_name: str,
    top_k: int = 5,
):
    encoder = CrossEncoder(model_name)
    for elem in inputs:
        elem["cross_encoder_score"] = encoder.predict([[query, elem["text"]]])[0].item()

    # sort the results byt the cross encoder score
    inputs = sorted(inputs, key=lambda x: x["cross_encoder_score"], reverse=True)

    return inputs[:top_k]


if __name__ == "__main__":
    print("running reranker")
    inputs = [
        {
            "text": "This is a test"
        },
        {
            "text": "This is the answer to the meaning of life"
        },
        {
            "text": "This is another text"
        },
        {
            "text": "2 + 2 is 4"
        }
    ]

    query = "What is a test"
    model_name = "distilroberta-base"
    top_k = 2

    reranked = re_ranker(
        inputs=inputs,
        query=query,
        model_name=model_name,
        top_k=top_k
    )
    import json
    print(json.dumps(reranked))