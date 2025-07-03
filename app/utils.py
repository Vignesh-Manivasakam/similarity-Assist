import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import plotly.express as px
from app.preprocess import preprocess_data
import logging

def excel_to_json(file_obj, tokenizer):
    """Convert Excel file object to JSON with preprocessing."""
    try:
        file_obj.seek(0)
        df = pd.read_excel(file_obj, dtype={'Object_Text': str})
        
        if 'Object_Identifier' not in df.columns or 'Object_Text' not in df.columns:
            raise ValueError("Excel must have 'Object_Identifier' and 'Object_Text' columns")
        
        df['Object_Text'] = df['Object_Text'].fillna('').astype(str)
        df['Object_Identifier'] = df['Object_Identifier'].fillna('').astype(str)
        
        logging.info(f"Excel file processed: {len(df)} rows")
        logging.debug(f"Excel columns: {df.columns.tolist()}")
        logging.debug(f"Sample data: {df.head(2).to_dict()}")
        
        data = df[['Object_Identifier', 'Object_Text']].to_dict(orient='records')
        
        processed_data, skipped_empty_count = preprocess_data(data, tokenizer)
        
        logging.info(f"Preprocessed {len(processed_data)} entries, skipped {skipped_empty_count} empty texts")
        return processed_data, skipped_empty_count
        
    except Exception as e:
        logging.error(f"Failed to process Excel file: {str(e)}")
        raise ValueError(f"Error processing Excel file: {str(e)}")

def plot_embeddings(base_embeddings, user_embeddings, base_data, user_data):
    """Generate 3D embedding visualization."""
    try:
        combined_embeddings = np.vstack((base_embeddings, user_embeddings))
        
        pca = PCA(n_components=3)
        reduced = pca.fit_transform(combined_embeddings)
        
        base_count = len(base_embeddings)
        user_count = len(user_embeddings)
        categories = ['Base'] * base_count + ['Query'] * user_count
        texts = [entry['Original_Text'] for entry in base_data] + [entry['Original_Text'] for entry in user_data]
        
        fig = px.scatter_3d(
            x=reduced[:, 0], 
            y=reduced[:, 1], 
            z=reduced[:, 2],
            color=categories, 
            color_discrete_map={'Base': '#00d4ff', 'Query': '#ffaa00'},
            hover_data={'text': texts, 'category': categories},
            title="3D Embedding Visualization (Click a point to view sentence)",
            labels={'x': 'Main Pattern', 'y': 'Secondary Pattern', 'z': 'Tertiary Pattern'},
            opacity=0.8,
            size=[10] * len(reduced),
            size_max=15
        )
        
        fig.update_traces(
            hovertemplate="%{customdata[0]}<extra></extra>",
            marker=dict(line=dict(width=0))
        )
        
        fig.update_layout(
            scene=dict(
                xaxis_title="Main Pattern", 
                yaxis_title="Secondary Pattern",
                zaxis_title="Tertiary Pattern", 
                bgcolor="#1e1e2f"
            ),
            paper_bgcolor="#1e1e2f", 
            font=dict(color="white"),
            clickmode='event+select', 
            legend=dict(title="Embedding Type", font=dict(color="white")),
            width=1000,
            height=640,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        return fig
        
    except Exception as e:
        logging.error(f"Error creating embedding plot: {str(e)}")
        fig = px.scatter_3d(title="Embedding Visualization (Error occurred)")
        return fig