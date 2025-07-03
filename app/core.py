import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import logging
import os
import hashlib
from app.postprocess import highlight_word_differences
from app.config import (
    HF_MODEL_NAME, BASE_EMBEDDINGS_FILE,
    FAISS_INDEX_FILE, HASH_FILE
)

logger = logging.getLogger(__name__)

def load_model():
    """
    Load the sentence transformer model from Hugging Face Hub.
    """
    try:
        logger.info(f"Loading model from Hugging Face Hub: {HF_MODEL_NAME}")
        model = SentenceTransformer(HF_MODEL_NAME)
        logger.info(f"Successfully loaded model: {HF_MODEL_NAME}")
        return model
    except Exception as e:
        logger.error(f"FATAL: Failed to load model from Hugging Face Hub. Error: {e}")
        raise RuntimeError(f"Could not load embedding model. Please check your internet connection. Details: {e}")

def _compute_file_hash(file_obj):
    """Compute SHA256 hash of a file-like object."""
    sha256 = hashlib.sha256()
    file_obj.seek(0)
    for chunk in iter(lambda: file_obj.read(4096), b""):
        sha256.update(chunk)
    file_obj.seek(0)
    return sha256.hexdigest()

def _check_cache_validity(current_hash: str) -> bool:
    """Checks if the cached files exist and the hash matches."""
    if not all(os.path.exists(f) for f in [BASE_EMBEDDINGS_FILE, FAISS_INDEX_FILE, HASH_FILE]):
        return False
    try:
        with open(HASH_FILE, "r", encoding='utf-8') as f:
            cached_hash = f.read().strip()
        if cached_hash == current_hash:
            logger.info("Cache is valid.")
            return True
        logger.info("Cache hash mismatch. Regenerating index.")
        return False
    except IOError as e:
        logger.warning(f"Error reading cache hash file: {e}")
        return False

def _load_from_cache():
    """Loads the FAISS index and embeddings from their cache files using secure methods."""
    try:
        logger.info("Loading FAISS index and embeddings from cache.")
        embeddings = np.load(BASE_EMBEDDINGS_FILE)
        index = faiss.read_index(FAISS_INDEX_FILE)
        return index, embeddings
    except Exception as e:
        logger.warning(f"Failed to load from cache: {e}. Regeneration will occur.")
        return None, None

def _generate_and_save_index(data: list, model, current_hash: str):
    """Generates new embeddings and FAISS index, then saves them to cache."""
    logger.info("Generating new embeddings and FAISS index.")
    texts = [entry['Cleaned_Text'] for entry in data]
    if not texts:
        raise ValueError("No texts found for embedding generation")

    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    normalized = normalize(embeddings, axis=1, norm='l2')
    dim = normalized.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(normalized)

    try:
        np.save(BASE_EMBEDDINGS_FILE, normalized)
        faiss.write_index(index, FAISS_INDEX_FILE)
        with open(HASH_FILE, "w", encoding='utf-8') as f:
            f.write(current_hash)
        logger.info("Successfully cached embeddings and index.")
    except (IOError, faiss.FaissException) as e:
        logger.warning(f"Failed to cache embeddings/index: {e}")

    return index, normalized

def create_faiss_index(data: list, model, base_file_obj):
    """
    Refactored to orchestrate cache checking, loading, and generation.
    It uses secure numpy methods instead of pickle for embeddings.
    """
    os.makedirs(os.path.dirname(FAISS_INDEX_FILE), exist_ok=True)
    current_hash = _compute_file_hash(base_file_obj)

    if _check_cache_validity(current_hash):
        index, embeddings = _load_from_cache()
        if index is not None and embeddings is not None:
             return index, embeddings, data

    index, embeddings = _generate_and_save_index(data, model, current_hash)
    return index, embeddings, data

def search_similar(user_data, index, base_data, top_k, thresholds, model):
    """Perform similarity search with FAISS and include word difference highlighting."""
    results = []
    user_texts = [entry['Cleaned_Text'] for entry in user_data]
    if not user_texts:
        raise ValueError("No user texts found for similarity search")

    user_embeddings = model.encode(user_texts, convert_to_numpy=True, show_progress_bar=True)
    user_embeddings = normalize(user_embeddings, axis=1, norm='l2')
    D, I = index.search(user_embeddings, top_k)

    for i, entry in enumerate(user_data):
        for rank in range(top_k):
            score = float(D[i][rank])
            match_idx = I[i][rank]

            if match_idx >= len(base_data):
                continue

            match = base_data[match_idx]
            if abs(score - 1.0) < 1e-6: label = "Exact Match"
            elif score >= thresholds['most']: label = "Most Similar"
            elif score >= thresholds['moderate']: label = "Moderately Similar"
            else: label = "No Match"

            query_words, match_words = highlight_word_differences(entry['Original_Text'], match['Original_Text'])
            results.append({
                'Query_Object_Identifier': entry['Object_Identifier'], 'Query_Sentence': entry['Original_Text'],
                'Query_Sentence_Cleaned_text': entry['Cleaned_Text'], 'Query_Sentence_Highlighted': query_words,
                'Matched_Object_Identifier': match['Object_Identifier'], 'Matched_Sentence': match['Original_Text'],
                'Matched_Sentence_Cleaned_text': match['Cleaned_Text'], 'Matched_Sentence_Highlighted': match_words,
                'Similarity_Score': round(score, 4), 'Similarity_Level': label
            })
    return results, user_embeddings