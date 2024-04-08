import os
import gradio as gr
from dotenv import load_dotenv
load_dotenv()
from utils import LlamaIndex
from retriever import VectorDBRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

def main():
    # Load the LlamaIndex
    li = LlamaIndex(
        db_name="vector_db",
        db_user=os.getenv("PG_USER"),
        db_password=os.getenv("PG_PASSWORD"),
        db_host=os.getenv("PG_HOST"),
        db_port=os.getenv("PG_PORT"),
        embed_model="jinaai/jina-embeddings-v2-base-zh"
    )
    li.load_data('./data/')
    retriever = VectorDBRetriever(li.vector_store, li.embed_model, similarity_top_k=5)
    query_engine = RetrieverQueryEngine.from_args(retriever, llm=li.llm)
    
    demo = gr.Interface(
        fn=query_engine.query,
        inputs=gr.Textbox(lines=7, label="Query"),
        outputs="text",
        title="RAG Demo",
        description="Ask me anything!"
    )
    
    demo.launch(share=True)
    
if __name__ == "__main__":
    main()