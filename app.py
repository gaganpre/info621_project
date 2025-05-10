import streamlit as st
import os
from rag import PDFQuestionAnswering
from graph_builder import visualize_interactive, extract_id
import plotly
import json
import traceback

# Streamlit page configuration
st.set_page_config(page_title="RAG Citation Graph Assistant", layout="wide")

# Initialize session state
if 'pdf_qa' not in st.session_state:
    st.session_state.pdf_qa = None
if 'papers_loaded' not in st.session_state:
    st.session_state.papers_loaded = False
if 'citation_graph' not in st.session_state:
    st.session_state.citation_graph = None
if 'graph_edges' not in st.session_state:
    st.session_state.graph_edges = None
if 'root_id' not in st.session_state:
    st.session_state.root_id = None
if 'processed_pdfs' not in st.session_state:
    st.session_state.processed_pdfs = set()

# Create papers directory if it doesn't exist
PAPERS_DIR = "./papers"
if not os.path.exists(PAPERS_DIR):
    os.makedirs(PAPERS_DIR)

def initialize_pdf_qa():
    """Initialize the PDFQuestionAnswering system."""
    try:
        st.session_state.pdf_qa = PDFQuestionAnswering(chunk_size=3000, chunk_overlap=0)
        # Load citation graph data
        with open("citations.json", "r") as f:
            citation_data = json.load(f)
        st.session_state.root_id = extract_id(list(citation_data.keys())[0])
        st.session_state.citation_graph, st.session_state.graph_edges = (
            st.session_state.pdf_qa.G, 
            st.session_state.pdf_qa.G.edges()
        )
        st.session_state.papers_loaded = False  # Set to False until PDFs are loaded
    except Exception as e:
        st.error(f"Error initializing system: {str(e)}")
        st.session_state.pdf_qa = None

# Sidebar for PDF upload and controls
st.sidebar.title("Controls")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF Papers", 
    type=["pdf"], 
    accept_multiple_files=True
)

if uploaded_files:
    with st.sidebar.status("Processing uploaded PDFs..."):
        try:
            # Initialize PDFQuestionAnswering if not already done
            if st.session_state.pdf_qa is None:
                initialize_pdf_qa()
            
            # Process each uploaded PDF only if not already processed
            for uploaded_file in uploaded_files:
                file_name = uploaded_file.name
                if file_name not in st.session_state.processed_pdfs:
                    file_path = os.path.join(PAPERS_DIR, file_name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    st.write(f"Processing {file_name}...")
                    documents = st.session_state.pdf_qa.load_document(file_path)
                    st.session_state.pdf_qa.add_documents(documents)
                    st.session_state.processed_pdfs.add(file_name)
                    st.write(f"Loaded {file_name}")
                else:
                    st.write(f"Skipping {file_name} (already processed)")
            
            st.session_state.papers_loaded = True
            st.success("All new PDFs processed successfully!")
        except Exception as e:
            st.error(f"Error processing PDFs: {str(e)}")
            st.session_state.papers_loaded = False

# Main content
st.title("RAG Citation Graph Assistant")
st.markdown("Upload academic papers and ask questions to get answers with citation context and explanations.")

# Display loaded PDFs
if st.session_state.processed_pdfs:
    st.markdown("**Loaded Papers**:")
    for pdf in st.session_state.processed_pdfs:
        st.markdown(f"- {pdf}")

# Query input
query = st.text_input("Enter your question:", placeholder="e.g., How is attention different from previous sequence models?")
submit_button = st.button("Submit Query")

if submit_button and query:
    if st.session_state.pdf_qa is None:
        st.error("Please upload at least one PDF to initialize the system.")
    elif not st.session_state.papers_loaded:
        st.error("No papers loaded. Please upload PDFs first.")
    else:
        with st.status("Processing query..."):
            try:
                # Query the system without reprocessing PDFs
                response = st.session_state.pdf_qa.ask_question(query)
                
                # Display answer
                st.subheader("Answer")
                st.markdown(response["answer"])
                
                # Display explanation
                st.subheader("Explanation")
                
                for chunk in response["explanation"]["retrieved_chunks"]:
                    st.markdown(f"**Content**: {chunk['content']}")
                    st.markdown(f"**Metadata**: {chunk['metadata']}")
                    st.markdown("---")
            
                st.markdown("**Citation Path**: " + " → ".join(response["explanation"]["citation_path"]))
                st.markdown(f"**Confidence**: {response['explanation']['confidence']}")
                
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")
                st.markdown("**Stack Trace** (for debugging):")
                st.code(traceback.format_exc())

# Graph visualization
st.subheader("Citation Graph")
if st.session_state.citation_graph is not None and st.session_state.root_id is not None:
    try:
        # Generate interactive graph
        fig = visualize_interactive(
            st.session_state.citation_graph, 
            st.session_state.graph_edges, 
            st.session_state.root_id
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error rendering citation graph: {str(e)}")
else:
    st.info("Upload PDFs to view the citation graph.")

# Instructions
st.sidebar.markdown("---")
st.sidebar.subheader("Instructions")
st.sidebar.markdown("""
1. Upload one or more PDF papers in the sidebar.
2. Enter a question related to the papers (e.g., "What is scaled dot-product attention?").
3. Click **Submit Query** to get an answer with explanations.
4. View the interactive citation graph below, showing relationships between papers.
""")

# Cleanup option
if st.sidebar.button("Clear Papers and Reset"):
    try:
        # shutil.rmtree(PAPERS_DIR)
        # os.makedirs(PAPERS_DIR)
        st.session_state.pdf_qa = None
        st.session_state.papers_loaded = False
        st.session_state.citation_graph = None
        st.session_state.graph_edges = None
        st.session_state.root_id = None
        st.session_state.processed_pdfs = set()
        st.sidebar.success("Papers cleared and system reset.")
    except Exception as e:
        st.sidebar.error(f"Error resetting: {str(e)}")