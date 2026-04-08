from dotenv import load_dotenv
import os

load_dotenv()

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.core import Settings, PromptTemplate
from langfuse import get_client


POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5433")
POSTGRES_DB = os.getenv("POSTGRES_DB", "ragapp")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "password")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
LITELLM_API_BASE = os.getenv("LITELLM_API_BASE", "http://localhost:4000")
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY", "")

# LangFuse picks up keys automatically from env after load_dotenv()
langfuse = get_client()

Settings.llm = OpenAILike(
    model="mistral",
    api_base=LITELLM_API_BASE,
    api_key=LITELLM_API_KEY,
    is_chat_model=True
)

Settings.embed_model = OllamaEmbedding(
    model_name=OLLAMA_EMBED_MODEL,
    base_url=OLLAMA_BASE_URL
)

PROMPT_TEMPLATE = PromptTemplate("""You are an assistant for ACME Corporation.
Use only the context below to answer the question.
If the answer is not in the context, say "I don't have that information."
Do not make up answers.

Context:
{context_str}

Question: {query_str}

Answer:""")

def load_index():
    vector_store = PGVectorStore.from_params(
        database=POSTGRES_DB,
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        table_name="document_embeddings",
        embed_dim=768
    )
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )
    return VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context
    )

def query(question: str):
    with langfuse.start_as_current_observation(
        as_type="span",
        name="rag-query",
        input=question
    ) as span:
        index = load_index()
        query_engine = index.as_query_engine(
            similarity_top_k=3,
            text_qa_template=PROMPT_TEMPLATE
        )

        with langfuse.start_as_current_observation(
            as_type="span",
            name="retrieval"
        ):
            response = query_engine.query(question)

        span.update(output=str(response))

    langfuse.flush()
    return response

if __name__ == "__main__":
    questions = [
        "How many days of annual leave do employees get?",
        "What is the process for production deployments?",
        "How often is each engineer on call?",
        "Can employees carry over unused sick leave?"
    ]

    for question in questions:
        print(f"\nQuestion: {question}")
        response = query(question)
        print(f"Answer: {response}")
        print("-" * 50)
