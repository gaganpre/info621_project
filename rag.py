from typing import List
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings

from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import chain
from langchain_core.runnables import RunnableLambda, RunnableParallel
from graph_builder import graph_retrieval, build_citation_graph
from operator import itemgetter
# from langchain_openai import OpenAI
# from langchain.chat_models import init_chat_model
# from langchain_openai import OpenAIEmbeddings

# embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# from langchain_openai import OpenAI
from langchain_ollama import ChatOllama
import getpass
import os
import simplejson as json
import numpy as np
import warnings

clean_title = lambda x : x.split("#")[2].strip()

from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType

# Suppress the specific tokenizer warning
warnings.filterwarnings("ignore", category=FutureWarning, 
                       message="`clean_up_tokenization_spaces` was not set")

try:
    # os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "http://localhost:6006"
    from phoenix.otel import register
        # configure the Phoenix tracer
    tracer_provider = register(
        project_name="my-llm-app", # Default is 'default'
        auto_instrument=True # Auto-instrument your app based on installed OI dependencies
    )
    print("INFO: Tracing enabled.")
except Exception as e:
    print(f"WARNING: Not able to enable Tracing: {e}")
    # Handle the error or log it as needed
class CustomEmbeddings(Embeddings): 
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        return [self.model.encode(d).tolist() for d in documents]

    def embed_query(self, query: str) -> List[float]:
        return self.model.encode([query])[0].tolist()


class PDFQuestionAnswering:
    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 200):
        # Initialize OpenAI LLM

        self.llm = ChatOllama(model="llama3.2:3b", temperature=0.1)

        # Initialize embedding model
        self.embedding_model = CustomEmbeddings(model_name="all-MiniLM-L6-v2")
        # self.embedding_model = embeddings
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
            # length_function=len
        )
        
        # Initialize empty vectorstore
        self.vectorstore = Chroma(
            embedding_function=self.embedding_model,
            persist_directory="chroma",
            collection_name="pdf_qa",
        )
        
        # Setup QA chain
        self.qa_chain = self._setup_qa_chain()
        with open("citations.json", "r") as _file:
            self.citation_data = json.load(_file)
        root_id = list(self.citation_data.keys())[0]
        self.G, _ = build_citation_graph(root_id, self.citation_data[root_id], root_title="Attention is All You Need")


    def load_documents_from_dir(self, directory: str):
        # Load documents from the specified directory
        document_loader = PyPDFDirectoryLoader(directory)
        documents = document_loader.load()
        processed_files_for_titles = set() # To avoid processing title for every page of the same file

        print("\n--- Extracted Titles from Directory ---")
        for doc in documents:
            source_file = doc.metadata.get('source')
            if source_file and source_file not in processed_files_for_titles:
                title = doc.metadata.get('title')
                base_filename = os.path.basename(source_file)
                if title:
                    print(f"File: {base_filename}, Title (from metadata): {title}")
                    self.extracted_titles[base_filename] = title
                else:
                    # Fallback: use filename (without extension) if title metadata is missing
                    filename_title = os.path.splitext(base_filename)[0].replace('_', ' ').replace('-', ' ')
                    print(f"File: {base_filename}, Title (from filename): {filename_title}")
                    self.extracted_titles[base_filename] = filename_title
                processed_files_for_titles.add(source_file)
        if not documents:
            print("No PDF documents found or loaded from the directory.")
        print("-------------------------------------\n")
        
        return documents

    def load_document(self, file_path: str):

        # title = extract_title_from_pdf(file_path)
        l = DoclingLoader(file_path, export_type=ExportType.MARKDOWN).load()

        # print(f"Extracted title: {}")
        title = clean_title(l[0].page_content.split("\n")[0])
        print("+==========================")
        print( "TITLE : ", title)
        print("+==========================")

        # document_loader = PyPDFLoader(file_path, extract_images=False)


        # documents = document_loader.load()
        # metadata={"title" : }

        for doc in l:
            doc.metadata.update({"title": title})

        # return documents
        return l

    def add_documents(self, documents, metadata={}):
        print(documents[0].metadata)
        if documents:
            # Split documents into chunks
            chunks = self.text_splitter.split_documents(documents)
            # Add chunks to vectorstore
            self.vectorstore.add_documents(chunks)
        else:
            raise Exception("No documents to add.")

    def ask_question_v1(self, question: str) -> str:
        # Ask a question and get the answer
        if not self.vectorstore:
            raise Exception("No documents in the vectorstore.")
        
        response = self.qa_chain.invoke({"question": question})
        return response

    def ask_question(self, question: str):
        if not self.vectorstore:
            raise Exception("No documents in the vectorstore.")
        
        # Get vector and graph results
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        vector_results = self.vectorstore.similarity_search_with_score(question, k=5)
        # Convert to expected format for fusion
        vector_results_formatted = {
            'documents': [[doc.page_content for doc, _ in vector_results]],
            'metadatas': [[doc.metadata for doc, _ in vector_results]],
            'distances': [[score for _, score in vector_results]]
        }
        query_embedding = self.embedding_model.embed_query(question)
        # self.G , edges = build_citation_graph(list(data.keys())[0], data[list(data.keys())[0]], root_title)
        graph_results = graph_retrieval(self.G, question, top_k=3, max_hops=2, embedding_model=self.embedding_model)
        
        # Fuse results
        fused_results = self.fuse_and_rank_results(vector_results_formatted, graph_results, query_embedding, top_n=5)
        
        # Prepare context for LLM
        context = "\n\n".join([res["content"] for res in fused_results])
        citation_path = [res["metadata"].get("title", res["content"].replace("Title: ", "")) 
                         for res in fused_results if res["source"] == "citation_graph"]
        
        # Generate answer
        response = self.qa_chain.invoke({"question": question, "context": context})
        
        # Compute confidence
        vector_scores = [res["relevance_score"] for res in fused_results if res["source"] == "vector_db"]
        graph_scores = [res["relevance_score"] for res in fused_results if res["source"] == "citation_graph"]
        avg_vector_score = np.mean(vector_scores) if vector_scores else 1
        avg_graph_score = np.mean(graph_scores) if graph_scores else 1
        confidence = 0.6 * avg_vector_score + 0.4 * avg_graph_score
        
        # Structure response
        return {
            "answer": response,
            "explanation": {
                "retrieved_chunks": [
                    {"content": res["content"], "metadata": res["metadata"]}
                    for res in fused_results if res["source"] == "vector_db"
                ],
                "citation_path": citation_path,
                "confidence": round(confidence, 2)
            }
        }

    # def _setup_qa_chain(self, root_title="Attention is all you need"):
    #     """Setup the QA chain using LCEL, incorporating fused retrieval."""

    #     # 1. Create the prompt template
    #     prompt = ChatPromptTemplate.from_messages([
    #         ("system", (
    #             "You are an assistant for question-answering tasks. "
    #             "Use the following pieces of retrieved context to answer "
    #             "the question. If you don't know the answer, say that you "
    #             "don't know. Use three sentences maximum and keep the "
    #             "answer concise."
    #             "\n\n"
    #             "<context>"
    #             "{context}"
    #             "</context>"
    #         )),
    #         ("human", "{question}")
    #     ])

    #     # 2. Setup base retriever from the vectorstore
    #     base_retriever = self.vectorstore.as_retriever()

    #     # 3. Setup graph retriever
    #     try:
    #         # Ensure citations.json is accessible.
    #         # You might need to adjust the path if it's not in the current working directory.
    #         with open("citations.json", "r") as _file:
    #             citation_data = json.load(_file)
            
    #         if not citation_data:
    #             print("Warning: citations.json is empty. Graph features might be limited.")
    #             # Fallback or specific handling if citation_data is crucial and empty
    #             first_paper_key, first_paper_citations = None, None
    #         else:
    #             first_paper_key = list(citation_data.keys())[0]
    #             first_paper_citations = citation_data[first_paper_key]
    #         from graph_builder import build_citation_graph, graph_retrieval
    #         if first_paper_key and first_paper_citations:
    #             G, _ = build_citation_graph(first_paper_key, first_paper_citations, root_title)
    #             # graph_retrieval MUST return a LangChain BaseRetriever or compatible Runnable
    #             graph_retriever_component = graph_retrieval(G, root_title)
    #         else:
    #             print("Warning: Could not get citation data for graph. Using dummy graph retriever.")
    #             graph_retriever_component = RunnableLambda(lambda x_query: []) # Returns no docs

    #     except FileNotFoundError:
    #         print("Warning: citations.json not found. Graph retriever will be disabled.")
    #         graph_retriever_component = RunnableLambda(lambda x_query: []) # Returns no docs
    #     except Exception as e:
    #         print(f"Warning: Error setting up graph retriever: {e}. Graph retriever will be disabled.")
    #         graph_retriever_component = RunnableLambda(lambda x_query: [])


    #     # 4. Define the chain segment for combined context generation
    #     # This part fetches from both retrievers and then fuses their results.
    #     # The input to this segment will be the original question string.
        
    #     # This will pass {"question": "the_query"} to the runnables
    #     # and expect them to handle this input structure if they are custom.
    #     # Standard retrievers handle the string query directly when piped from itemgetter("question").
        
    #     # Step 4.1: Retrieve documents from both sources in parallel
    #     # The input to this parallel step is the dict {"question": "user_query"}
    #     parallel_retrieval = RunnableParallel(
    #         docs_base=itemgetter("question") | base_retriever,
    #         docs_graph=itemgetter("question") | graph_retriever_component,
    #         original_question=itemgetter("question") # Pass question through for fusion logic
    #     )
        
    #     # Step 4.2: Define the fusion logic as a RunnableLambda
    #     # This lambda will receive the output of parallel_retrieval:
    #     # a dict like {"docs_base": [...], "docs_graph": [...], "original_question": "..."}
    #     def fusion_lambda_function(retrieved_data: dict):
    #         base_docs = retrieved_data["docs_base"]
    #         graph_docs = retrieved_data["docs_graph"]
    #         query_str = retrieved_data["original_question"]
            
    #         # Call your core fusion and ranking logic here
    #         # This `user_core_fusion_function` should encapsulate the logic
    #         # previously in `fuse_and_rank_results(retriever, _graph_retriever)`
    #         # but now operating on the *results* of retrievals and the query.
    #         fused_docs = user_core_fusion_function(base_docs, graph_docs, query=query_str)
    #         return fused_docs

    #     # Combined context generation chain:
    #     # Input to this chain: {"question": "user_query"}
    #     # Output of this chain: List[Document] (fused documents)
    #     fused_context_chain = parallel_retrieval | RunnableLambda(fusion_lambda_function)


    #     # 5. Assemble the final QA chain
    #     # The overall chain expects an input like {"question": "user's question"}
    #     final_qa_chain = (
    #         {
    #             # The 'context' key for the prompt will be populated by fused_context_chain
    #             # It needs the original {"question": "..."} input.
    #             "context": fused_context_chain,
    #             # The 'question' key for the prompt
    #             "question": itemgetter("question")
    #         }
    #         | prompt
    #         | self.llm
    #         | StrOutputParser()
    #     )
        
    #     return final_qa_chain

    def fuse_and_rank_results(self, vector_results, graph_results, query_embedding, top_n=5):
        fused_results = []
        # Process vector results
        for i in range(len(vector_results['documents'][0])):
            distance = vector_results['distances'][0][i]
            relevance_score = 1 - distance  # Convert cosine distance to similarity (0-1)
            fused_results.append({
                "source": "vector_db",
                "content": vector_results['documents'][0][i],
                "metadata": vector_results['metadatas'][0][i],
                "relevance_score": relevance_score
            })
        # Process graph results
        for paper in graph_results:
            # Compute semantic similarity between query and node label
            label = paper.get('label', '')
            label_embedding = self.embedding_model.embed_query(label)
            similarity = np.dot(query_embedding, label_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(label_embedding))
            # Add citation count as a factor (normalize to 0-1)
            cited_by_count = self.citation_data.get(paper['id'], {}).get('cited_by_count', 0)
            citation_score = min(cited_by_count / 100, 1.0)  # Cap at 100 citations
            # Combine scores (weighted)
            relevance_score = 0.7 * similarity + 0.3 * citation_score
            fused_results.append({
                "source": "citation_graph",
                "content": f"Title: {label}",
                "metadata": {"openalex_id": paper.get('id'), "type": paper.get('type')},
                "relevance_score": relevance_score
            })
        # Rank and select top-N
        ranked_results = sorted(fused_results, key=lambda x: x['relevance_score'], reverse=True)
        return ranked_results[:top_n]

    def _setup_qa_chain(self, root_title="Attention is All You Need"):
        prompt = ChatPromptTemplate.from_messages([
            ("system", ("You are an assistant for question-answering tasks. "
                        "Use the following pieces of retrieved context to answer "
                        "the question. If you don't know the answer, say that you "
                        "don't know. Use three sentences maximum and keep the "
                        "answer concise."
                        "\n\n"
                        "<context>"
                        "{context}"
                        "</context>")),
            ("human", "{question}")
        ])
        
        chain = (
            {"context": lambda x: x["context"], "question": lambda x: x["question"]}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return chain

    def _setup_qa_chain_v1(self, root_title="Attention is all you need"):
        """Setup the QA chain with Claude."""
        
        
        # Create the prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", ("You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer "
                "the question. If you don't know the answer, say that you "
                "don't know. Use three sentences maximum and keep the "
                "answer concise."
                "\n\n"
                "<context>"
                "{context}"
                "</context>")),
            ("human", "{question}")
        ])
        
        # Setup retrieval chain using the new LCEL interface
        retriever = self.vectorstore.as_retriever()
        print(retriever)
        # import sys
        # sys.exit() 

        # # citation data
        with open("citations.json", "r") as _file:
            data = json.load(_file)

        G, edges = build_citation_graph(list(data.keys())[0], data[list(data.keys())[0]], root_title)
        
        _graph_retriever = graph_retrieval(G, root_title)
        print(_graph_retriever)
        # results = fuse_and_rank_results(
        #     retriever,
        #     _graph_retriever
        # )

        print("Fused results:")
        # print(results)



        chain = (
             
            {"context": retriever, "question": lambda x: x}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return chain


if __name__ == "__main__":
    # if not os.environ.get("OPENAI_API_KEY"):
    #     os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")
    # _E = CustomEmbeddings(model_name="all-MiniLM-L6-v2")


    # Example usage
    pdf_qa = PDFQuestionAnswering(chunk_size=3000, chunk_overlap=0)
    
    # Load documents from a directory
    # documents = pdf_qa.load_documents_from_dir("./papers")
    
    # Load each pdf file separately
    # if not os.path.exists("./chroma"):
    papers_dir = "./papers"
    import glob
    pdf_files = glob.glob(os.path.join(papers_dir, "*.pdf"))
    for document in pdf_files:
        print(f"Processing file: {document}")
        print("==========================")
        # print(f"Loading document: {document}")
        documents= pdf_qa.load_document(document)
        # Add documents to the vectorstore

        pdf_qa.add_documents(documents)
        
    question = input("Enter your question: ")
    # Ask a question
    answer = pdf_qa.ask_question(question)
    print(answer)

    # question = input("Enter your question: ")
    # answer = pdf_qa.ask_question(question)
    print("Answer:", answer["answer"])
    print("\nExplanation:")
    print("Retrieved Chunks:")
    for chunk in answer["explanation"]["retrieved_chunks"]:
        print(f"- {chunk['content']} (Metadata: {chunk['metadata']})")
    print("Citation Path:", " → ".join(answer["explanation"]["citation_path"]))
    print(f"Confidence: {answer['explanation']['confidence']}")