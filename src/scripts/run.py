import os
from promptflow import PFClient


_rag_flow_path = os.path.join(os.path.dirname(__file__), "..", "flows", "rag-flow")
_evaluation_flow_path = os.path.join(os.path.dirname(__file__), "..", "flows", "evaluation", "evaluation-flow")
_data_path = os.path.join(_rag_flow_path, "data.jsonl")


def main():
    pf_client = PFClient()

    print("Running RAG flow...")
    flow_run = pf_client.run(
        flow=_rag_flow_path,
        data=_data_path)
    pf_client.stream(flow_run)

    print("Running evaluation flow...")
    eval_run = pf_client.run(
        flow=_evaluation_flow_path,
        data=_data_path,
        column_mapping={
            'groundtruth': "${data.expected_output}",
            'prediction': "${run.outputs.answer}"
        },
        run=flow_run
    )
    pf_client.stream(eval_run)

    # getting the evaluation results
    metrics = pf_client.get_metrics(eval_run)
    print("Evaluation results:")
    print(metrics)


if __name__ == "__main__":
    main()
