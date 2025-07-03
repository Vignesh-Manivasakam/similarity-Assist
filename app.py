import os
import logging
from logging.handlers import RotatingFileHandler
import sys
import pandas as pd
import streamlit as st
# Set page config as the first Streamlit command
st.set_page_config(layout="wide", page_title="Requivalance")

from app.config import (
    LOG_LEVEL, DEFAULT_THRESHOLDS, MAX_FILE_SIZE,
    LLM_ANALYSIS_MIN_THRESHOLD, LLM_PERFECT_MATCH_THRESHOLD
)
from app.pipeline import run_similarity_pipeline
from app.postprocess import display_summary, create_highlighted_excel, df_to_html_table
from app.utils import plot_embeddings
from app.llm_service import get_llm_analysis_batch, client as llm_client  # Import llm_client

# Setup logging
for handler in logging.root.handlers[:]:
    logging.root.handlers.remove(handler)

log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'app.log')

file_handler = RotatingFileHandler(
    log_file, mode='a', maxBytes=5*1024*1024, backupCount=5, encoding='utf-8'
)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logging.getLogger('').setLevel(LOG_LEVEL.upper())
logging.getLogger('').addHandler(file_handler)
logging.getLogger('').addHandler(console_handler)

def load_sidebar():
    """Manages the Streamlit sidebar for file uploads and settings."""
    with st.sidebar:
        st.header("📂 File Selection & Settings")
        with st.expander("Upload Files", expanded=True):
            st.markdown('<div class="card">', unsafe_allow_html=True)
            
            # --- Base File Logic ---
            base_file = st.file_uploader("Base Excel File 📑", type=["xlsx"])
            if base_file:  # This is the crucial check
                if base_file.size > MAX_FILE_SIZE:
                    st.error(f"Base file size exceeds {MAX_FILE_SIZE/1024/1024}MB limit")
                    # We invalidate the file by setting it to None so the rest of the app knows
                    base_file = None 
                else:
                    # Only show success if the file is valid
                    st.markdown('<p class="upload-success">✅ Base file uploaded!</p>', unsafe_allow_html=True)

            # --- Check File Logic ---
            check_file = st.file_uploader("Check Excel File 📝", type=["xlsx"])
            if check_file: # This is the crucial check
                if check_file.size > MAX_FILE_SIZE:
                    st.error(f"Check file size exceeds {MAX_FILE_SIZE/1024/1024}MB limit")
                    # We invalidate the file by setting it to None
                    check_file = None
                else:
                    # Only show success if the file is valid
                    st.markdown('<p class="upload-success">✅ Check file uploaded!</p>', unsafe_allow_html=True)

            top_k = st.number_input("Top K Matches 🎯", min_value=1, max_value=10, value=3)
            st.markdown('</div>', unsafe_allow_html=True)

        with st.expander("Similarity Thresholds", expanded=True):
            st.markdown('<div class="card">', unsafe_allow_html=True)
            thresholds = {
                'exact': st.slider("Exact Match Threshold", 0.9, 1.0, DEFAULT_THRESHOLDS['exact'], 0.001),
                'most': st.slider("Most Similar Threshold", 0.7, 0.9, DEFAULT_THRESHOLDS['most'], 0.01),
                'moderate': st.slider("Moderately Similar Threshold", 0.5, 0.7, DEFAULT_THRESHOLDS['moderate'], 0.01)
            }
            st.markdown('</div>', unsafe_allow_html=True)

        run_btn = st.button("🚀 Run Similarity Search")
        
    return base_file, check_file, top_k, thresholds, run_btn

def main():
    """Main function to run the Streamlit application."""
    st.markdown("<h1 style='text-align: center;'>📘 Requivalance — 🧠Smart Requirement Matcher🔍</h1>", unsafe_allow_html=True)
    try:
        with open("static/css/custom.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("Custom CSS file not found. Using default styling.")

    base_file, check_file, top_k, thresholds, run_btn = load_sidebar()

    if 'results_df' not in st.session_state:
        st.session_state.results_df = None
    if 'total_tokens' not in st.session_state:
        st.session_state.total_tokens = {'prompt_tokens': 0, 'completion_tokens': 0}

    if run_btn:
        if not base_file or not check_file:
            st.warning("⚠️ Please upload both Excel files.")
            return

        with st.spinner("Running initial similarity search with FAISS..."):
            try:
                results, base_embeddings, user_embeddings, base_data, user_data, base_skipped_count, user_skipped_count = run_similarity_pipeline(
                    base_file, check_file, top_k, thresholds, st.empty()
                )
                st.session_state.results_df = pd.DataFrame(results)
                st.session_state.base_data = base_data
                st.session_state.user_data = user_data
                st.session_state.base_embeddings = base_embeddings
                st.session_state.user_embeddings = user_embeddings
                st.session_state.stats = (base_skipped_count, user_skipped_count)
                st.session_state.total_tokens = {'prompt_tokens': 0, 'completion_tokens': 0}
                st.success("✅ FAISS Search Complete! Results are shown below.")
            except Exception as e:
                st.error(f"❌ An error occurred during the pipeline: {e}")
                logging.error(f"Pipeline error: {e}", exc_info=True)
                st.session_state.results_df = None

    if st.session_state.results_df is not None:
        df = st.session_state.results_df
        base_data = st.session_state.base_data
        user_data = st.session_state.user_data
        base_skipped, user_skipped = st.session_state.stats

        summary_col, vis_col = st.columns(2)
        with vis_col:
            st.subheader("📊 Embedding Visualization")
            fig = plot_embeddings(st.session_state.base_embeddings, st.session_state.user_embeddings, base_data, user_data)
            st.plotly_chart(fig, use_container_width=True)
        with summary_col:
            st.subheader("📈 Summary")
            display_summary(df.to_dict('records'))

        st.subheader("🤖 AI-Powered Deep Analysis")
        if llm_client is None:
            st.warning("⚠️ LLM key and URL is not added. LLM analysis is unavailable.")
        else:
            st.markdown(f"""
                Click the button below to use a powerful LLM to analyze the results.
                - Pairs with `Similarity Score` > **{LLM_PERFECT_MATCH_THRESHOLD}** will be auto-labeled 'Equivalent (Auto)'.
                - Pairs with `Similarity Score` < **{LLM_ANALYSIS_MIN_THRESHOLD}** will be skipped.
                """)

            if st.button("🔬 Enhance with LLM Analysis"):
                if 'LLM_Relationship' in df.columns:
                    st.info("LLM analysis has already been performed.")
                else:
                    progress_bar = st.progress(0, text="Starting LLM analysis...")
                    llm_results = []
                    total_tokens = {'prompt_tokens': 0, 'completion_tokens': 0}
                    total_rows = len(df)
                    batch_size = 10

                    for i in range(0, total_rows, batch_size):
                        batch_rows = df.iloc[i:i + batch_size]
                        sentence_pairs = [
                            (row['Query_Sentence_Cleaned_text'], row['Matched_Sentence_Cleaned_text'])
                            for _, row in batch_rows.iterrows()
                        ]
                        batch_results = [None] * len(sentence_pairs)
                        llm_pairs = []
                        llm_indices = []

                        for j, (sentence1, sentence2) in enumerate(sentence_pairs):
                            score = batch_rows.iloc[j]['Similarity_Score']
                            if score >= LLM_PERFECT_MATCH_THRESHOLD:
                                batch_results[j] = {'LLM_Score': 1.0, 'LLM_Relationship': 'Equivalent (Auto)'}
                            elif score < LLM_ANALYSIS_MIN_THRESHOLD:
                                batch_results[j] = {'LLM_Score': 'N/A', 'LLM_Relationship': 'Below Threshold'}
                            else:
                                llm_pairs.append((sentence1, sentence2))
                                llm_indices.append(j)

                        if llm_pairs:
                            batch_response = get_llm_analysis_batch(llm_pairs)
                            llm_batch_results = batch_response['results']
                            total_tokens['prompt_tokens'] += batch_response['tokens_used'].get('prompt_tokens', 0)
                            total_tokens['completion_tokens'] += batch_response['tokens_used'].get('completion_tokens', 0)
                            logging.info(f"Batch {i//batch_size + 1}: Prompt tokens = {batch_response['tokens_used']['prompt_tokens']}, Completion tokens = {batch_response['tokens_used']['completion_tokens']}, LLM Pairs = {len(llm_pairs)}, LLM Results = {len(llm_batch_results)}")

                            if len(llm_batch_results) != len(llm_pairs):
                                logging.error(f"Batch {i//batch_size + 1}: LLM result length mismatch: expected {len(llm_pairs)}, got {len(llm_batch_results)}")
                                st.error(f"Error: Batch {i//batch_size + 1} LLM results length mismatch.")
                                return

                            for idx, result in zip(llm_indices, llm_batch_results):
                                batch_results[idx] = result

                        if None in batch_results:
                            logging.error(f"Batch {i//batch_size + 1}: Missing results for some indices: {batch_results}")
                            st.error(f"Error: Batch {i//batch_size + 1} has missing results.")
                            return

                        if len(batch_results) != len(sentence_pairs):
                            logging.error(f"Batch {i//batch_size + 1} result length mismatch: expected {len(sentence_pairs)}, got {len(batch_results)}")
                            st.error(f"Error: Batch {i//batch_size + 1} produced incorrect number of results.")
                            return

                        llm_results.extend(batch_results)
                        progress_bar.progress(min((i + batch_size) / total_rows, 1.0), text=f"Analyzing batch {i//batch_size + 1}...")

                    if len(llm_results) != total_rows:
                        logging.error(f"LLM results length mismatch: expected {total_rows}, got {len(llm_results)}")
                        st.error(f"Error: LLM results length ({len(llm_results)}) does not match expected ({total_rows}).")
                        return

                    llm_df = pd.DataFrame(llm_results, index=df.index)
                    st.session_state.results_df = pd.concat([df, llm_df], axis=1)
                    st.session_state.total_tokens = total_tokens

                    st.success("✅ LLM Analysis Complete!")
                    st.rerun()

        st.subheader("📋 Results Table")
        note = f"**Note**: Skipped {base_skipped} base and {user_skipped} query entries due to empty or invalid text."
        if 'LLM_Relationship' in df.columns and llm_client is not None:
            total_token_count = st.session_state.total_tokens.get('prompt_tokens', 0) + st.session_state.total_tokens.get('completion_tokens', 0)
            note += f"<br>📊 **Total LLM tokens used**: {total_token_count} ({st.session_state.total_tokens.get('prompt_tokens', 0)} prompt + {st.session_state.total_tokens.get('completion_tokens', 0)} completion)"
        st.markdown(note, unsafe_allow_html=True)
        st.markdown(df_to_html_table(df, base_data, user_data), unsafe_allow_html=True)

        st.subheader("📥 Download Results")
        st.markdown("Download the results table, including any LLM analysis, in your preferred format.")
        download_container = st.container()
        with download_container:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="download-card">', unsafe_allow_html=True)
                st.download_button(
                    label="📊 Download Excel Results",
                    data=create_highlighted_excel(df),
                    file_name="similarity_results_highlighted.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="excel_download"
                )
                st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="download-card">', unsafe_allow_html=True)
                st.download_button(
                    label="💾 Download JSON Results",
                    data=df.to_json(orient="records", indent=2),
                    file_name="similarity_results.json",
                    mime="application/json",
                    key="json_download"
                )
                st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
