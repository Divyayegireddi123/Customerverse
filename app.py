# app.py
import streamlit as st
import pandas as pd
import io

from pypdf import PdfReader
import docx

from text_processor import clean_and_tokenize
from vectorizer import CustomTFIDFVectorizer
from clustering import AgglomerativeClusteringFromScratch, calculate_cosine_distance
from evaluation import calculate_silhouette_score

st.set_page_config(page_title="Hierarchical Document Clustering Engine", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .report-title { font-size:2.4rem; font-weight:700; color:#1E3A8A; margin-bottom:0.2rem; }
    .report-subtitle { font-size:1.1rem; color:#4B5563; margin-bottom:1.5rem; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='report-title'>Hierarchical Document Clustering System</div>", unsafe_allow_html=True)
st.markdown("<div class='report-subtitle'>An unsupervised machine learning pipeline for text categorization implemented from fundamental engineering principles.</div>", unsafe_allow_html=True)

st.sidebar.header("⚙️ System Parameters")
linkage_criterion = st.sidebar.selectbox("Linkage Matrix Criterion", ["Complete Linkage (Max Distance)"])

def extract_text_from_file(uploaded_file):
    filename = uploaded_file.name.lower()
    try:
        if filename.endswith('.txt'):
            return uploaded_file.read().decode("utf-8", errors="ignore")
        elif filename.endswith('.pdf'):
            pdf_bytes = io.BytesIO(uploaded_file.read())
            pdf_reader = PdfReader(pdf_bytes)
            extracted_text = [page.extract_text() for page in pdf_reader.pages if page.extract_text()]
            return " ".join(extracted_text)
        elif filename.endswith('.docx'):
            docx_bytes = io.BytesIO(uploaded_file.read())
            doc_object = docx.Document(docx_bytes)
            return " ".join([paragraph.text for paragraph in doc_object.paragraphs])
    except Exception as error_msg:
        st.sidebar.error(f"Failed parsing text from {uploaded_file.name}: {str(error_msg)}")
        return ""
    return ""

col_input, col_analytics = st.columns([1, 1.2])

with col_input:
    st.subheader("1. Document Ingestion Module")
    uploaded_files = st.file_uploader("Upload target documents for algorithmic clustering:", type=["txt", "pdf", "docx"], accept_multiple_files=True)
    execute_pipeline = st.button("Execute Clustering Pipeline", type="primary", disabled=not uploaded_files)
    if not uploaded_files:
        st.info("💡 Drag and drop 2 or more files (.txt, .pdf, or .docx) above to run the pipeline.")

with col_analytics:
    st.subheader("2. Algorithmic Analytics Dashboard")
    if execute_pipeline and uploaded_files:
        documents, document_names = [], []
        for file_obj in uploaded_files:
            file_text = extract_text_from_file(file_obj)
            if file_text.strip():
                documents.append(file_text.strip())
                document_names.append(file_obj.name)
                
        if len(documents) < 2:
            st.error("Validation Error: System requires a minimum of 2 valid files containing readable text layers.")
        else:
            tokenized_corpus = [clean_and_tokenize(doc) for doc in documents]
            vectorizer = CustomTFIDFVectorizer()
            tfidf_matrix = vectorizer.fit_transform(tokenized_corpus)
            engine = AgglomerativeClusteringFromScratch()
            merge_history = engine.fit(tfidf_matrix)
            
            st.markdown("### Matrix Diagnostics")
            m_col1, m_col2 = st.columns(2)
            m_col1.metric(label="Corpus Size (N)", value=f"{len(documents)} Files")
            m_col2.metric(label="Feature Dimensionality (V)", value=f"{len(vectorizer.vocabulary)} Unique Terms")
            
            st.session_state['pipeline_processed'] = True
            st.session_state['docs'] = documents
            st.session_state['doc_names'] = document_names
            st.session_state['tokens'] = tokenized_corpus
            st.session_state['matrix'] = tfidf_matrix
            st.session_state['vocab'] = vectorizer.vocabulary
            st.session_state['history'] = merge_history

if st.session_state.get('pipeline_processed'):
    docs, doc_names, tokens, matrix, history = st.session_state['docs'], st.session_state['doc_names'], st.session_state['tokens'], st.session_state['matrix'], st.session_state['history']
    
    st.markdown("---")
    tab_tokens, tab_distance, tab_hierarchy, tab_cut = st.tabs(["🔬 Tokenization View", "📊 Pairwise Distance Matrix", "🌳 Dendrogram Trace", "✂️ Dynamic Tree Segmentation"])
    
    with tab_tokens:
        st.markdown("### Text Processing Matrix View")
        token_data = []
        for i, (name, toke) in enumerate(zip(doc_names, tokens)):
            preview = docs[i][:150] + "..." if len(docs[i]) > 150 else docs[i]
            token_data.append({"File Index": f"File {i}", "Filename": name, "Raw Preview": preview, "Extracted Word Tokens": str(toke[:15]) + "..."})
        st.table(token_data)

    with tab_distance:
        st.markdown("### Normalized Pairwise Cosine Distance Matrix")
        column_headers = [f"File {i} ({name[:12]}...)" if len(name) > 12 else f"File {i} ({name})" for i, name in enumerate(doc_names)]
        matrix_rows = []
        for i in range(len(docs)):
            row = []
            for j in range(len(docs)):
                row.append(0.0000 if i == j else float(f"{calculate_cosine_distance(matrix[i], matrix[j]):.4f}"))
            matrix_rows.append(row)
        st.dataframe(pd.DataFrame(matrix_rows, columns=column_headers, index=column_headers), use_container_width=True)

    with tab_hierarchy:
        st.markdown("### 🌳 Algorithmic Tree Representation")
        dot = ["digraph Tree {", "    node [shape=box, style=filled, fillcolor=\"#F3F4F6\", fontname=\"Helvetica\", color=\"#1E3A8A\"];", "    edge [color=\"#9CA3AF\", penwidth=1.5];", "    rankdir=BT;"]
        for i, name in enumerate(doc_names):
            dot.append(f"    {i} [label=\"File {i}\\n({name.replace('\"', '\\\"')})\", fillcolor=\"#DBEAFE\", color=\"#1E3A8A\"];")
        for step_entry in history:
            c1, c2 = step_entry['merged_nodes']
            res_id = step_entry['new_node_id']
            dot.append(f"    {res_id} [label=\"Cluster {res_id}\\nLinkage: {step_entry['linkage_distance']:.3f}\", fillcolor=\"#FEF3C7\", color=\"#D97706\"];")
            dot.append(f"    {c1} -> {res_id};")
            dot.append(f"    {c2} -> {res_id};")
        dot.append("}")
        st.graphviz_chart("\n".join(dot))

    with tab_cut:
        st.markdown("### Dynamic Cluster Selection & Performance Review")
        target_clusters = st.slider("Target Number of Extracted Categories:", min_value=1, max_value=len(docs), value=min(3, len(docs)))
        steps_to_take = len(docs) - target_clusters
        current_clusters = {i: [i] for i in range(len(docs))}
        for i in range(steps_to_take):
            c1, c2 = history[i]['merged_nodes']
            current_clusters[history[i]['new_node_id']] = current_clusters[c1] + current_clusters[c2]
            del current_clusters[c1], current_clusters[c2]
            
        indiv_scores, global_avg_score = calculate_silhouette_score(matrix, current_clusters, doc_names)
        eval_col1, eval_col2 = st.columns([1, 1.2])
        with eval_col1:
            st.metric(label="Overall System Silhouette Score", value=f"{global_avg_score:.4f}")
            if global_avg_score > 0.5: st.success("Analysis conclusion: High quality cluster separation confirmed.")
            elif global_avg_score > 0.0: st.warning("Analysis conclusion: Weak overlap detected.")
            else: st.error("Analysis conclusion: Overlapping structures.")
        with eval_col2:
            st.dataframe([{"File Name Target": k, "Silhouette Quality": f"{v:.4f}"} for k, v in indiv_scores.items()], use_container_width=True)
            
        st.markdown("---")
        c_cols = st.columns(len(current_clusters))
        for idx, (cluster_key, document_indices) in enumerate(current_clusters.items()):
            with c_cols[idx]:
                st.markdown(f"##### 📁 Category Folder {idx + 1}")
                st.caption(f"Cluster Identifier: {cluster_key}")
                for doc_id in document_indices:
                    st.info(f"📄 **{doc_names[doc_id]}** (File {doc_id})")
