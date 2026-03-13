import os
# --- เพิ่มส่วนนี้: ตั้งค่า Cache ไปยัง Drive D ก่อน import library อื่นๆ ---
os.environ['HF_HOME'] = 'D:/huggingface_cache'
os.environ['HUGGINGFACE_HUB_CACHE'] = 'D:/huggingface_cache'
import os
token = os.getenv("HF_TOKEN")

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
import torch


# --- เพิ่มส่วนนี้: สั่ง Login (ระบบจะข้ามไปถ้าเคย Login ในเครื่องนี้แล้ว) ---
# หรือใช้ login(token="your_token_here") ถ้าไม่อยากพิมพ์ใน terminal
from huggingface_hub import login
login(token=token)

def load_vectorstore(index_dir):
    """Loads the FAISS index with SentencePiece embeddings."""
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    
    # Allow dangerous deserialization is required in recent langchain updates 
    # for loading local pickle files securely
    vectorstore = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
    return vectorstore

def load_llm():
    """Loads a lightweight Instruct model (Qwen 1.5B) to prevent Out-Of-Memory crashes."""
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    
    print(f"Loading tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    print(f"Loading model {model_id} (Saving to Drive D)...")
    
    # เพิ่มการใช้ 4-bit quantization เพื่อประหยัด VRAM (Optional)
    # หากมีการ์ดจอแรงๆ (24GB+) สามารถเอา load_in_4bit ออกได้
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map="auto",
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        # cache_dir="D:/huggingface_cache" # ระบุซ้ำเพื่อความชัวร์
    )
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.3, # Slightly higher temperature to prevent getting stuck
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.1, # Prevent repeating "I don't know"
        return_full_text=False
    )
    
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

# Define the RAG prompt template based on Qwen's ChatML format
prompt_template = """<|im_start|>system
You are a helpful movie expert assistant. Your job is to answer the user's question using the provided movie plot context. 
If the information is in the context, synthesize it into a clear answer.
If the answer is truly not present in the context, politely state that you cannot answer based on the given information.<|im_end|>
<|im_start|>user
Here is the movie plot context:
{context}

Based on the context above, answer the following question:
{question}<|im_end|>
<|im_start|>assistant
"""

PROMPT = PromptTemplate(
    template=prompt_template, 
    input_variables=["context", "question"]
)

def answer_question(question, vectorstore, llm):
    """
    Retrieves the most relevant plot chunks for a question,
    sends them to the LLM, and returns the generated answer 
    along with the source documents.
    """
    # Retrieve top 3 most relevant chunks with their similarity scores (L2 distance in FAISS)
    # Note: For FAISS L2, lower score is better (more similar)
    docs_and_scores = vectorstore.similarity_search_with_score(question, k=3)
    
    contexts = []
    sources = []
    
    for doc, score in docs_and_scores:
        contexts.append(doc.page_content)
        sources.append({
            "title": doc.metadata.get("title", "Unknown"),
            "genre": doc.metadata.get("genre", "Unknown"),
            "year": doc.metadata.get("year", "Unknown"),
            "origin": doc.metadata.get("origin", "Unknown"),
            "director": doc.metadata.get("director", "Unknown"),
            "cast": doc.metadata.get("cast", "Unknown"),
            "wiki_page": doc.metadata.get("wiki_page", "Unknown"),
            "score": float(score),
            "content": doc.page_content
        })
        
    combined_context = "\n\n".join(contexts)
    
    # Format the prompt
    final_prompt = PROMPT.format(context=combined_context, question=question)
    
    # Generate the answer using the LLM with a terminator to stop after answering
    print(f"\nGenerating answer for: '{question}'...")
    
    terminators = [
        "<|im_end|>", 
        "<|endoftext|>"
    ]
    
    answer = llm.invoke(final_prompt, stop=terminators)
    
    return {
        "answer": answer.strip(),
        "sources": sources
    }

if __name__ == "__main__":
    import sys
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    index_dir = os.path.join(base_dir, "data", "faiss_index")
    
    if not os.path.exists(index_dir):
        print(f"Error: Vector index not found at {index_dir}. Run build_index.py first.")
        sys.exit(1)
        
    print("Loading resources...")
    vectorstore = load_vectorstore(index_dir)
    llm = load_llm()
    
    # Test Question
    test_q = "What happens at the end of Titanic?"
    result = answer_question(test_q, vectorstore, llm)
    
    print("\n" + "="*50)
    print(f"Question: {test_q}")
    print(f"Answer: {result['answer']}")
    print("="*50)
    print("\nSources:")
    for i, src in enumerate(result['sources']):
        print(f"{i+1}. Movie: {src['title']} (Score: {src['score']:.4f})")
