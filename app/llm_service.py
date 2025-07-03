import os
import json
import logging
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import requests
import hashlib
import tiktoken
from app.config import (
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_CACHE_FILE,
    SYSTEM_PROMPT_PATH
)

logger = logging.getLogger(__name__)

def initialize_client():
    """Initialize OpenAI client if API key and base URL are provided."""
    if not LLM_API_KEY or not LLM_BASE_URL:
        logger.warning("LLM API Key or Base URL not configured.")
        return None
    try:
        client = OpenAI(
            api_key="dummy",
            base_url=LLM_BASE_URL,
            default_headers={"": LLM_API_KEY},
            timeout=30
        )
        logger.info("OpenAI client initialized successfully.")
        return client
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
        return None

client = initialize_client()

# Load system prompt from external file for maintainability
try:
    with open(SYSTEM_PROMPT_PATH, 'r', encoding='utf-8') as f:
        SYSTEM_PROMPT_TEMPLATE = f.read()
except FileNotFoundError:
    logger.error(f"System prompt file not found at: {SYSTEM_PROMPT_PATH}")
    SYSTEM_PROMPT_TEMPLATE = "Error: System prompt could not be loaded."

def compute_prompt_hash(prompt: str) -> str:
    """Compute a hash for a prompt."""
    return hashlib.sha256(prompt.encode('utf-8')).hexdigest()

def load_llm_cache():
    """Load cached LLM results using safe JSON."""
    if not os.path.exists(LLM_CACHE_FILE):
        return {}
    try:
        with open(LLM_CACHE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Failed to load or parse LLM cache '{LLM_CACHE_FILE}': {e}. Starting with an empty cache.")
        return {}

def save_llm_cache(cache):
    """Save LLM results to cache using safe JSON."""
    try:
        os.makedirs(os.path.dirname(LLM_CACHE_FILE), exist_ok=True)
        with open(LLM_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=4)
    except (TypeError, IOError) as e:
        logger.error(f"Failed to save LLM cache: {e}")

def estimate_tokens(text: str) -> int:
    """Estimate token count using tiktoken for better accuracy."""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text, disallowed_special=()))
    except Exception:
        # Fallback for when tiktoken might fail
        return len(text) // 4

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((requests.exceptions.RequestException,))
)
def _call_llm_api(full_prompt: str, num_pairs: int) -> dict:
    """Helper function to make the actual API call and handle responses."""
    content, prompt_tokens, completion_tokens = "", 0, 0
    try:
        prompt_tokens = estimate_tokens(full_prompt)
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": full_prompt}],
            temperature=0.0,
            n=1
        )
        content = response.choices[0].message.content
        completion_tokens = estimate_tokens(content)
        json_str = content.strip().lstrip('```json').rstrip('```').strip()
        analysis = json.loads(json_str)

        if not isinstance(analysis, list) or len(analysis) != num_pairs:
            raise ValueError(f"LLM response length mismatch. Expected {num_pairs}, got {len(analysis)}")

        results = [
            {'LLM_Score': result.get('score', 'Error'), 'LLM_Relationship': result.get('relationship', 'Parse Error')}
            for result in analysis
        ]
        return {'results': results, 'tokens_used': {'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens}}

    except (requests.exceptions.RequestException, json.JSONDecodeError, ValueError) as e:
        error_type = type(e).__name__
        logger.error(f"LLM call failed due to {error_type}: {e}. Response: '{content}'")
        error_msg = f"{error_type} Error"
        return {'results': [{'LLM_Score': 'Error', 'LLM_Relationship': error_msg}] * num_pairs, 'tokens_used': {'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens}}
    except Exception as e:
        logger.error(f"An unexpected LLM error occurred: {e}. Response: '{content}'")
        return {'results': [{'LLM_Score': 'Error', 'LLM_Relationship': 'LLM Call Failed'}] * num_pairs, 'tokens_used': {'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens}}

def _process_batch(batch_pairs: list, cache: dict) -> dict:
    """Processes a single batch of sentence pairs against the LLM, using caching."""
    batch_prompt_body = ""
    for i, (s1, s2) in enumerate(batch_pairs):
        batch_prompt_body += f"Pair {i+1}:\nSentence 1: \"{s1}\"\nSentence 2: \"{s2}\"\n"

    full_prompt = SYSTEM_PROMPT_TEMPLATE + "\n" + batch_prompt_body
    cache_key = compute_prompt_hash(full_prompt)

    if cache_key in cache:
        logger.info(f"Returning cached LLM result for batch key: {cache_key[:10]}...")
        return cache[cache_key]

    logger.info(f"Calling LLM for a batch of {len(batch_pairs)} pairs.")
    response = _call_llm_api(full_prompt, len(batch_pairs))
    
    # Only cache successful, valid results
    if all(res.get('LLM_Relationship') not in ['Error', 'Parse Error'] for res in response['results']):
        cache[cache_key] = response
        
    return response

def get_llm_analysis_batch(sentence_pairs: list) -> dict:
    """
    Analyzes sentence pairs by iteratively creating batches that respect token limits.
    Returns an error if the client is not configured.
    """
    if not client:
        return {'results': [{'LLM_Score': 'Error', 'LLM_Relationship': 'LLM URL and Key not available'}] * len(sentence_pairs), 'tokens_used': {}}

    cache = load_llm_cache()
    all_results = [None] * len(sentence_pairs)
    total_tokens_used = {'prompt_tokens': 0, 'completion_tokens': 0}

    base_prompt_tokens = estimate_tokens(SYSTEM_PROMPT_TEMPLATE)
    current_batch_pairs = []
    current_batch_indices = []
    current_batch_tokens = base_prompt_tokens

    for i, pair in enumerate(sentence_pairs):
        pair_text = f"Pair {len(current_batch_pairs)+1}:\nSentence 1: \"{pair[0]}\"\nSentence 2: \"{pair[1]}\"\n"
        pair_tokens = estimate_tokens(pair_text)
        
        if current_batch_pairs and current_batch_tokens + pair_tokens > LLM_BATCH_TOKEN_LIMIT:
            batch_response = _process_batch(current_batch_pairs, cache)
            for j, res in enumerate(batch_response['results']):
                all_results[current_batch_indices[j]] = res
            
            total_tokens_used['prompt_tokens'] += batch_response['tokens_used'].get('prompt_tokens', 0)
            total_tokens_used['completion_tokens'] += batch_response['tokens_used'].get('completion_tokens', 0)
            
            current_batch_pairs, current_batch_indices, current_batch_tokens = [], [], base_prompt_tokens

        current_batch_pairs.append(pair)
        current_batch_indices.append(i)
        current_batch_tokens += pair_tokens

    if current_batch_pairs:
        batch_response = _process_batch(current_batch_pairs, cache)
        for j, res in enumerate(batch_response['results']):
            all_results[current_batch_indices[j]] = res
            
        total_tokens_used['prompt_tokens'] += batch_response['tokens_used'].get('prompt_tokens', 0)
        total_tokens_used['completion_tokens'] += batch_response['tokens_used'].get('completion_tokens', 0)
    
    save_llm_cache(cache)
    
    # Sanity check to ensure no results were missed
    if None in all_results:
        logger.error("LLM analysis resulted in missing data points. Filling with errors.")
        all_results = [res if res is not None else {'LLM_Score': 'Error', 'LLM_Relationship': 'Processing Error'} for res in all_results]

    return {'results': all_results, 'tokens_used': total_tokens_used}