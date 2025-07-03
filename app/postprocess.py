import pandas as pd
import streamlit as st
import plotly.express as px
from openpyxl import Workbook
import io
import logging
import difflib

def df_to_html_table(df, base_data, user_data):
    """Generates a styled HTML table from the results DataFrame with truncation indicators and colored LLM relationships."""
    html = '<table class="results-table">'
    html += '<tr style="background-color: #444; color: white; text-align: left;">'
    visible_columns = [
        'Query_Object_Identifier',
        'Query_Sentence_Highlighted',
        'Matched_Object_Identifier',
        'Matched_Sentence_Highlighted',
        'Similarity_Score',
        'Similarity_Level',
        'LLM_Score',
        'LLM_Relationship'
    ]
    
    for col in visible_columns:
        header_text = col.replace("_Highlighted", "").replace("_", " ").title()
        html += f'<th class="sentence-column" style="width: 12.5%;">{header_text}</th>'
    html += '</tr>'

    for _, row in df.iterrows():
        html += '<tr>'
        for col in visible_columns:
            if col == 'Query_Sentence_Highlighted':
                query_id = row['Query_Object_Identifier']
                is_truncated = next((entry['Truncated'] for entry in user_data if entry['Object_Identifier'] == query_id), False)
                text = f"<span style='color:yellow'>⚠️</span> {row[col]}" if is_truncated else row[col]
                html += f'<td class="sentence-column">{text}</td>'
            elif col == 'Matched_Sentence_Highlighted':
                match_id = row['Matched_Object_Identifier']
                is_truncated = next((entry['Truncated'] for entry in base_data if entry['Object_Identifier'] == match_id), False)
                text = f"<span style='color:yellow'>⚠️</span> {row[col]}" if is_truncated else row[col]
                html += f'<td class="sentence-column">{text}</td>'
            elif col == 'LLM_Relationship':
                relationship = row.get(col, 'N/A')
                color = '#4CAF50' if 'Equivalent' in str(relationship) else '#ff6b6b' if 'Contradictory' in str(relationship) else 'inherit'
                html += f'<td style="color: {color}; font-weight: bold;">{relationship}</td>'
            elif col == 'LLM_Score':
                score = row.get(col, 'N/A')
                html += f'<td>{score}</td>'
            else:
                html += f'<td>{str(row[col])}</td>'
        html += '</tr>'
    
    html += '</table>'
    return html

def highlight_word_differences(query_text, matched_text):
    """Highlight word differences between query and matched sentences for HTML display."""
    try:
        query_words = query_text.split()
        matched_words = matched_text.split()
        
        diff = list(difflib.ndiff(query_words, matched_words))
        
        highlighted_query = []
        highlighted_matched = []
        
        for d in diff:
            if d.startswith('  '):
                word = d[2:]
                highlighted_query.append(word)
                highlighted_matched.append(word)
            elif d.startswith('- '):
                word = d[2:]
                highlighted_query.append(f'<span style="color:#ff6b6b; font-weight:bold">{word}</span>')
            elif d.startswith('+ '):
                word = d[2:]
                highlighted_matched.append(f'<span style="color:#00d4ff; font-weight:bold">{word}</span>')
        
        return ' '.join(highlighted_query), ' '.join(highlighted_matched)
        
    except Exception as e:
        logging.error(f"Error in highlight_word_differences: {str(e)}")
        return query_text, matched_text

def create_highlighted_excel(df):
    """Generate a clean Excel file with selected similarity results."""
    try:
        wb = Workbook()
        ws = wb.active
        ws.title = "Similarity Results"
        
        headers = [
            'Query Id',
            'Query Sentence',
            'Matched Id',
            'Matched Sentence',
            'Similarity Score',
            'Similarity Level',
            'LLM Score',
            'LLM Relationship'
        ]
        ws.append(headers)
        
        for _, row in df.iterrows():
            ws.append([
                row['Query_Object_Identifier'],
                row['Query_Sentence'],
                row['Matched_Object_Identifier'],
                row['Matched_Sentence'],
                row['Similarity_Score'],
                row['Similarity_Level'],
                row.get('LLM_Score', 'N/A'),
                row.get('LLM_Relationship', 'N/A')
            ])
        
        for col in ws.columns:
            max_length = 0
            column = col[0].column_letter
            for cell in col:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)
            ws.column_dimensions[column].width = adjusted_width
        
        output = io.BytesIO()
        wb.save(output)
        output.seek(0)
        return output.getvalue()
        
    except Exception as e:
        logging.error(f"Error creating Excel file: {str(e)}")
        raise

def display_summary(results):
    """Display summary statistics and bar chart."""
    try:
        df = pd.DataFrame(results)
        
        if df.empty:
            st.warning("No results to display summary for.")
            return
        
        num_queries = len(df['Query_Object_Identifier'].unique())
        num_base = len(df['Matched_Object_Identifier'].unique())
        avg_similarity = df['Similarity_Score'].mean()
        level_counts = df['Similarity_Level'].value_counts().to_dict()
        
        st.markdown('<div class="summary-card">', unsafe_allow_html=True)
        st.markdown("**Summary Statistics**")
        st.write(f"- **Queries Processed**: {num_queries}")
        st.write(f"- **Base Sentences Matched**: {num_base}")
        st.write(f"- **Average Similarity Score**: {avg_similarity:.4f}")
        
        if level_counts:
            level_df = pd.DataFrame.from_dict(level_counts, orient='index', columns=['Count'])
            level_df = level_df.reset_index().rename(columns={'index': 'Similarity Level'})
            
            fig = px.bar(
                level_df, 
                x='Similarity Level', 
                y='Count', 
                title="Distribution of Similarity Levels",
                color='Similarity Level', 
                color_discrete_sequence=['#00d4ff', '#ffaa00', '#00ffaa', '#ff6b6b']
            )
            fig.update_layout(
                paper_bgcolor="#1e1e2f", 
                plot_bgcolor="#2a2a3c", 
                font=dict(color="white"),
                xaxis_title="Similarity Level", 
                yaxis_title="Count"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
    except Exception as e:
        logging.error(f"Error displaying summary: {str(e)}")
        st.error(f"Error displaying summary: {str(e)}")