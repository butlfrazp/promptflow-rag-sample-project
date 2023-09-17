import promptflow
from promptflow.connections import (
    CognitiveSearchConnection,
    AzureOpenAIConnection
)
from .utils.config import config


_COGNITIVE_SEARCH_CONNECTION_NAME = "cognitive_search_connection"
_AZURE_OPENAI_CONNECTION_NAME = "azure_openai_connection"


def main(
    cognitive_search_endpoint: str,
    cognitive_search_key: str,
    azure_openai_key: str,
    azure_openai_endpoint: str
):
    cognitive_search_connection = CognitiveSearchConnection(
        api_base=cognitive_search_endpoint,
        api_key=cognitive_search_key,
        name=_COGNITIVE_SEARCH_CONNECTION_NAME
    )

    azure_openai_connection = AzureOpenAIConnection(
        api_base=azure_openai_endpoint,
        api_key=azure_openai_key,
        name=_AZURE_OPENAI_CONNECTION_NAME
    )

    pf_client = promptflow.PFClient()
    pf_client.connections.create_or_update(cognitive_search_connection)
    pf_client.connections.create_or_update(azure_openai_connection)


if __name__ == "__main__":
    cognitive_search_endpoint = config.cognitive_search_endpoint
    cognitive_search_key = config.cognitive_search_api_key
    azure_openai_key = config.azure_openai_api_key
    azure_openai_endpoint = config.azure_openai_endpoint

    print("Creating connections...")

    main(
        cognitive_search_endpoint=cognitive_search_endpoint,
        cognitive_search_key=cognitive_search_key,
        azure_openai_key=azure_openai_key,
        azure_openai_endpoint=azure_openai_endpoint
    )
