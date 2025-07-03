import os
import logging
import json
import streamlit as st
from app.utils import excel_to_json
from app.config import OUTPUT_DIR
from app.core import load_model as core_load_model
from app.core import create_faiss_index, search_similar # Moved imports to top

@st.cache_resource
def get_model():
    """A Streamlit-cached wrapper to load the model only once."""
    logging.info("get_model() called. Caching model resource.")
    return core_load_model()

# Load the model once using the cached function
model = get_model()
tokenizer = model.tokenizer

def run_similarity_pipeline(base_file, check_file, top_k, thresholds, log_box):
    """Run the end-to-end similarity search pipeline with improved structure."""
    try:
        log_box.text("[INFO] Preprocessing files...")
        base_data, base_skipped = excel_to_json(base_file, tokenizer)
        user_data, user_skipped = excel_to_json(check_file, tokenizer)
        
        if not base_data or not user_data:
            raise ValueError("No valid data after preprocessing. Check uploaded files and logs.")
        
        log_box.text(f"[INFO] Building FAISS index for {len(base_data)} base items...")
        index, base_embeddings, base_data = create_faiss_index(base_data, model, base_file)
        
        log_box.text(f"[INFO] Performing similarity search for {len(user_data)} query items...")
        results, user_embeddings = search_similar(user_data, index, base_data, top_k, thresholds, model)
        
        log_box.text("[INFO] Processing complete!")
        logging.info(f"Pipeline completed successfully. Generated {len(results)} results.")
        
        return results, base_embeddings, user_embeddings, base_data, user_data, base_skipped, user_skipped
    except Exception as e:
        error_msg = f"Pipeline error: {e}"
        logging.error(error_msg, exc_info=True)
        log_box.text(f"[ERROR] {error_msg}")
        raise