
# Graph-RAG Powered Academic Assistant

This repository hosts the complete codebase, configuration, and documentation for the **Graph-RAG Powered Academic Assistant**. The assistant enhances standard Retrieval-Augmented Generation (RAG) systems with citation graph understanding and explainability features, providing users with trusted, citation-aware answers to academic questions.

---

## 🧠 Overview
The project aims to solve a common challenge in academia: understanding and tracing references in research papers. Our assistant integrates a semantic vector search engine with a citation graph constructed from sources like OpenAlex, enabling:

- Accurate retrieval of relevant research content
- Graph-based contextual expansion using paper citations
- Explainable answers with traceable citation paths

---

## 🚀 Features

### Core Capabilities
- 📄 PDF ingestion and semantic chunking
- 📚 Citation graph construction using OpenAlex API
- 🧭 Dual Retrieval: Vector similarity + Citation traversal
- 🧩 Fusion engine with weighted scoring
- 🗣️ LLM integration via Ollama (LLaMA 3.2B)
- 🔍 Explainability with citation trails and chunk-level trace
- 📊 Evaluation via BLEU/ROUGE and hallucination rate
- 🌐 Simple interactive UI (Streamlit)

### Technologies Used
| Component | Technology |
|----------|------------|
| LLM      | LLaMA 3.2B via Ollama |
| Embeddings | SentenceTransformer (MiniLM) |
| Vector DB | ChromaDB |
| Citation Graph | NetworkX + OpenAlex |
| PDF Parsing | PyMuPDF, Docling |
| UI       | Streamlit |
| Tracing  | Phoenix (OpenInference) |

---

## 📁 Project Structure
```
📂 graph-rag-academic-assistant
│
├── data_collector.py         # OpenAlex integration to fetch citation metadata
├── graph_builder.py          # Builds and visualizes citation graphs
├── rag.py                    # Main pipeline: embedding, retrieval, fusion, response
├── requirements.txt          # All required Python dependencies
├── Dockerfile                # Environment definition for container
├── docker-compose.yaml       # Multi-service orchestration: Ollama, App, Phoenix
├── 📂 papers/                # Folder to place input PDF files
├── 📂 docs/                  # Folder for project reports and other documents  
├── 📂 chroma/                # ChromaDB persistence
├── citations.json            # Cached citation metadata (auto-generated)
└── README.md                 # Project documentation
```

---

## 📦 Installation
### 1. Clone the repository
```bash
git clone https://github.com/gaganpre/info621_project.git
cd graph-rag-academic-assistant
```

### 2. OPTION 1:  Set up the environment with Docker (Automated Build) - Suggested
Make sure you have Docker and Docker Compose installed.
```bash
docker-compose up
```

NOTE: In case the build fails, run the command again :)

This will launch:
- The main app on [localhost:8501](http://localhost:8501)
- Ollama model service (LLaMA 3.2B)
- Phoenix for tracing at port 6006

---


### Option 2: Settting up manually 

```bash 
python -m venv project_env

source project_env/bin/activate

pip install -r requirements.txt


```

#### Running a model using ollama
```bash 
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama      
docker exec -it ollama ollama run llama3.2:3b
```


#### Enable Tracing (Optional)

```bash
docker pull arizephoenix/phoenix
docker run -p 6006:6006 -p 4317:4317 -i -t arizephoenix/phoenix:latest

```


#### Run Streamlit 

```bash 

streamlit run app.py

```


## 📥 Usage Guide


- Upload and parse documents
- Ask natural language questions
- View AI-generated answers + citation paths
- Visualize graph context and tracing metadata

---

## 📈 Evaluation Metrics
| Metric              | Value  |
|---------------------|--------|
| BLEU                | 0.71   |
| ROUGE-L             | 0.68   |
| Citation Accuracy   | 87%    |
| Hallucination Rate  | 6%     |
| Confidence Avg      | ~0.82  |
| Explainability Score| 4.6/5  |

---

## 📊 Visual Outputs
- Chunk Length Distribution Histogram
- Confidence Score Histogram
- Fusion Score Contribution (stacked bar)
- Citation Graph Visual (static and interactive)

All visualizations can be accessed from `graph_builder.py` or automatically through the app.

---

## 🧮 Confidence Score Formula
```text
Confidence = 0.6 × AvgVectorScore + 0.4 × AvgGraphScore
```
This formula weighs retrieval paths to ensure higher trust in the generated output.

---

## 📌 Sample Query Flow
**User Uploads:** "Attention is All You Need"

**Query:** "How does attention differ from earlier sequence models?"

**Assistant Response:**
> “Attention allows the model to focus on all tokens simultaneously, enabling parallel computation. This contrasts with RNN-based models. It builds upon Bahdanau (2015) and Luong (2015), both cited in the paper.”

**Graph Path:** Vaswani → Bahdanau → Luong

**Confidence:** 0.89

---

## 🔬 Future Work
- Support large-scale ingestion with Neo4j
- Fine-tuning domain-specific LLMs
- Web UI with search filtering & citation heatmaps
- Integration with educational platforms (Moodle, Canvas)

---

---

## 🙌 Acknowledgements
- [OpenAlex](https://openalex.org)
- [LangChain](https://www.langchain.com)
- [Ollama](https://ollama.ai)
- [Phoenix Tracing](https://arize.com/phoenix)
- [Hugging Face Transformers](https://huggingface.co)

For academic use or contributions, please raise an issue or contact us via GitHub.

---

**Developed for INFO 621 — AI Systems Project, Spring 2025**

Team 9 — Grape GPU 🍇
