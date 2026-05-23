from pathlib import Path

from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

ROOT_DIR = Path("E:/proyecto")

embeddings = OpenAIEmbeddings(
    model="text-embedding-qwen3-embedding-0.6b",
    base_url="http://localhost:1234/v1",
    api_key="lm-studio-dummy",
    check_embedding_ctx_length=False
)

vector_store = Chroma(
    persist_directory=str(ROOT_DIR / "chroma_db"),
    embedding_function=embeddings
)

@tool
def search_plant_info(query: str, species_name: str) -> str:
    """
    Busca información en el corpus de plantas canarias.
    Usa "species_name" cuando el usuario mencione una especie concreta o el modelo identifique una.
    """

    docs = vector_store.similarity_search(
        query=query,
        k=10,
        filter={"species_name": species_name}
    )

    if not docs:
        return "No se encontró información relevante en el corpus."

    return "\n\n".join(
        f"===\nSpecies: {doc.metadata.get('species_name')}\n{doc.page_content}\n"
        for doc in docs
    )
