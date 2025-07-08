import os

# Project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- File and Cache Paths ----
CACHE_DIR = os.path.join(BASE_DIR, 'data', 'cache')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'output')
PROMPT_DIR = os.path.join(BASE_DIR, 'app', 'prompts')

# Ensure directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PROMPT_DIR, exist_ok=True)

# Cache files
LLM_CACHE_FILE = os.path.join(CACHE_DIR, "llm_results_cache.json")
BASE_EMBEDDINGS_FILE = os.path.join(CACHE_DIR, "base_embeddings.npy")
FAISS_INDEX_FILE = os.path.join(CACHE_DIR, "base_index.faiss")
HASH_FILE = os.path.join(CACHE_DIR, "base_file_hash.txt")

# Model configuration
HF_MODEL_NAME = 'intfloat/e5-large-v2'
LLM_MODEL = 'meta-llama/Meta-Llama-3-70B-Instruct'

# Prompt file for maintainability
SYSTEM_PROMPT_PATH = os.path.join(PROMPT_DIR, 'system_prompt.txt')

# ---- Thresholds and Limits ----
DEFAULT_THRESHOLDS = {
    'exact': 1.0,
    'most': 0.9,
    'moderate': 0.7
}

# File size limit (10MB)
MAX_FILE_SIZE = 10 * 1024 * 1024

# Centralized configuration for scattered constants
MAX_TOKENS_FOR_TRUNCATION = 512
LLM_BATCH_TOKEN_LIMIT = 100000

# ---- Logging ----
LOG_LEVEL = 'INFO'

# ---- LLM Configuration ----
LLM_API_TOKEN = os.getenv("HF_TOKEN", "")
LLM_ANALYSIS_MIN_THRESHOLD = 0.80
LLM_PERFECT_MATCH_THRESHOLD = 0.999
