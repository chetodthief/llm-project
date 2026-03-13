import os
import json
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def build_vector_index(input_json, output_dir):
    print(f"Loading chunks from {input_json}...")
    with open(input_json, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
        
    texts = [chunk['text'] for chunk in chunks]
    metadatas = [chunk['metadata'] for chunk in chunks]
    
    # Initialize the embedding model requested
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"Loading embedding model: {model_name}...")
    
    # Note: HuggingFaceEmbeddings uses sentence-transformers under the hood
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    
    print("Building FAISS index (this may take a moment)...")
    # FAISS.from_texts generates embeddings for the texts and builds the index
    vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    
    # Save the index locally
    os.makedirs(output_dir, exist_ok=True)
    vectorstore.save_local(output_dir)
    print(f"Index successfully saved to {output_dir}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_file = os.path.join(base_dir, "data", "chunked_plots.json")
    output_dir = os.path.join(base_dir, "data", "faiss_index")
    build_vector_index(input_file, output_dir)
