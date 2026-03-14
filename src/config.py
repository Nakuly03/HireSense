import os
from dotenv import load_dotenv

load_dotenv()

def _require(key: str) -> str:
    val = os.getenv(key)
    if not val:
        raise EnvironmentError(f"Missing required env var: {key}")
    return val


HF_API_TOKEN: str = _require("HF_API_TOKEN")
HF_LLM_MODEL: str = os.getenv("HF_LLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct")


EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIM: int = 384


PINECONE_API_KEY: str = _require("PINECONE_API_KEY")
PINECONE_INDEX: str   = os.getenv("PINECONE_INDEX", "resumes-index")
PINECONE_CLOUD: str   = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION: str  = os.getenv("PINECONE_REGION", "us-east-1")


LANGSMITH_API_KEY: str | None = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT: str        = os.getenv("LANGSMITH_PROJECT", "hiresense")

if LANGSMITH_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"]    = LANGSMITH_API_KEY
    os.environ["LANGCHAIN_PROJECT"]    = LANGSMITH_PROJECT


API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8000"))


CATEGORIES: list[str] = [
    "HR", "DESIGNER", "INFORMATION-TECHNOLOGY", "TEACHER", "ADVOCATE",
    "BUSINESS-DEVELOPMENT", "HEALTHCARE", "FITNESS", "AGRICULTURE", "BPO",
    "SALES", "CONSULTANT", "DIGITAL-MEDIA", "AUTOMOBILE", "CHEF", "FINANCE",
    "APPAREL", "ENGINEERING", "ACCOUNTANT", "CONSTRUCTION",
    "PUBLIC-RELATIONS", "BANKING", "ARTS", "AVIATION",
]
