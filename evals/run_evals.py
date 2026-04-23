from dotenv import load_dotenv
import os
import json
import sys
load_dotenv()

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.core import Settings, PromptTemplate
from langfuse import get_client

langfuse = get_client()

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5433")
POSTGRES_DB = os.getenv("POSTGRES_DB", "ragapp")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "password")
LITELLM_API_BASE = os.getenv("LITELLM_API_BASE", "http://localhost:4000")
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY", "")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EVAL_THRESHOLD = float(os.getenv("EVAL_THRESHOLD", "0.7"))

Settings.llm = OpenAILike(
    model="mistral",
    api_base=LITELLM_API_BASE,
    api_key=LITELLM_API_KEY,
    is_chat_model=True
)
Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
    base_url=OLLAMA_BASE_URL
)

PROMPT_TEMPLATE = PromptTemplate("""You are an assistant for ACME Corporation.
Use only the context below to answer the question.
If the answer is not in the context, say "I don't have that information."

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

def judge_answer(question: str, expected: str, actual: str) -> float:
    judge_prompt = f"""You are evaluating an AI assistant's answer.

Question: {question}
Expected answer contains: {expected}
Actual answer: {actual}

Score the actual answer from 0 to 1:
- 1.0: Answer is correct and contains the expected information
- 0.5: Answer is partially correct
- 0.0: Answer is wrong, missing key information, or says it doesn't know when it should

Respond with ONLY a number between 0 and 1. Nothing else."""

    from llama_index.core.llms import ChatMessage
    import re

    response = Settings.llm.chat([
        ChatMessage(role="user", content=judge_prompt)
    ])

    # DEBUG - see exactly what's coming back
    raw = str(response)
    print(f"  RAG actual: '{actual[:80]}'")
    print(f"  Judge raw response: '{raw}'")

    # Extract number from response using regex
    # handles "0.8", "I give it 0.8", "Score: 1.0" etc
    matches = re.findall(r'\b(0\.\d+|1\.0|0\.0|0|1)\b', raw)
    if matches:
        score = float(matches[0])
        return min(max(score, 0.0), 1.0)

    print(f"  Could not parse score, defaulting to 0.0")
    return 0.0

def run_evals():
    print("Loading eval dataset...")
    with open("evals/eval_dataset.json") as f:
        dataset = json.load(f)

    print(f"Running {len(dataset)} evaluations...\n")
    index = load_index()
    query_engine = index.as_query_engine(
        similarity_top_k=3,
        text_qa_template=PROMPT_TEMPLATE
    )

    scores = []
    results = []

    for item in dataset:
        question = item["question"]
        expected = item["expected"]

        # Get answer from RAG
        response = query_engine.query(question)
        actual = str(response)

        # Judge the answer
        score = judge_answer(question, expected, actual)
        scores.append(score)

        result = {
            "question": question,
            "expected": expected,
            "actual": actual,
            "score": score,
            "passed": score >= EVAL_THRESHOLD
        }
        results.append(result)

        status = "PASS" if score >= EVAL_THRESHOLD else "FAIL"
        print(f"[{status}] Score: {score:.2f} | Q: {question[:50]}")

        # Log each eval to LangFuse
        with langfuse.start_as_current_observation(
            as_type="span",
            name="eval",
            input={"question": question, "expected": expected}
        ) as span:
            span.update(
                output={"actual": actual, "score": score}
            )

    langfuse.flush()

    # Summary
    avg_score = sum(scores) / len(scores)
    passed = sum(1 for s in scores if s >= EVAL_THRESHOLD)

    print(f"\n{'='*50}")
    print(f"Results: {passed}/{len(dataset)} passed")
    print(f"Average score: {avg_score:.2f}")
    print(f"Threshold: {EVAL_THRESHOLD}")

    # Save results for CI to read
    with open("eval_results.json", "w") as f:
        json.dump({
            "avg_score": avg_score,
            "passed": passed,
            "total": len(dataset),
            "threshold": EVAL_THRESHOLD,
            "results": results
        }, f, indent=2)

    # Exit with error code if below threshold
    # This is what makes the CI pipeline fail
    if avg_score < EVAL_THRESHOLD:
        print(f"\nFAILED: Average score {avg_score:.2f} below threshold {EVAL_THRESHOLD}")
        sys.exit(1)
    else:
        print(f"\nPASSED: Average score {avg_score:.2f} meets threshold {EVAL_THRESHOLD}")
        sys.exit(0)

if __name__ == "__main__":
    run_evals()
