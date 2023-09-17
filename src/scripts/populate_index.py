import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.azuresearch import AzureSearch
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from .utils.config import config


_INDEX_NAME = "tutorial-index"
_EMBEDDING_MODEL = "text-embedding-ada-002"

document_path = os.path.join(os.path.dirname(__file__), "documents/context.md")


def main():
    loader = TextLoader(document_path)

    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    # Create a vector store
    embeddings: OpenAIEmbeddings = OpenAIEmbeddings(
        deployment=_EMBEDDING_MODEL,
        chunk_size=1)
    vector_store = AzureSearch(
        azure_search_endpoint=config.cognitive_search_endpoint,
        azure_search_key=config.cognitive_search_api_key,
        index_name=_INDEX_NAME,
        embedding_function=embeddings.embed_query)
    vector_store.add_documents(docs)


if __name__ == "__main__":
    os.environ["OPENAI_API_TYPE"] = "azure"
    os.environ["OPENAI_API_BASE"] = config.azure_openai_endpoint
    os.environ["OPENAI_API_KEY"] = config.azure_openai_api_key
    os.environ["OPENAI_API_VERSION"] = "2023-05-15"

    print("Creating index...")
    main()
