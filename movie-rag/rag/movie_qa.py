import os
import sys
import pickle
import torch
from dotenv import load_dotenv

# --- 1. ตั้งค่า Cache และ Environment Variables ---
os.environ['HF_HOME'] = 'D:/huggingface_cache'
os.environ['HUGGINGFACE_HUB_CACHE'] = 'D:/huggingface_cache'

# โหลดตัวแปรจากไฟล์ .env (เพื่อซ่อน Token)
load_dotenv()
token = os.getenv("HF_TOKEN")

# --- 2. Imports Library ทั้งหมด ---
from huggingface_hub import login
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from langchain_core.prompts import PromptTemplate
from langchain_classic.retrievers.ensemble import EnsembleRetriever

# สั่ง Login ด้วย Token ที่ซ่อนไว้
if token:
    login(token=token)
else:
    print("Warning: HF_TOKEN not found in .env file!")

def load_vectorstore(index_dir):
    """โหลด FAISS และ BM25 เพื่อทำงานร่วมกันแบบ Hybrid Search"""
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    
    vectorstore = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    bm25_path = os.path.join(index_dir, "bm25_retriever.pkl")
    with open(bm25_path, "rb") as f:
        bm25_retriever = pickle.load(f)
        
    bm25_retriever.k = 5
    
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5]
    )
    return ensemble_retriever

def load_llm():
    """โหลด Llama 3 ด้วย 4-bit Quantization เพื่อประหยัด VRAM"""
    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    print(f"Loading tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    print(f"Loading model {model_id} (Saving to Drive D)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map="auto",
        quantization_config=bnb_config,
    )
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512, # เพิ่ม token เผื่อให้อธิบายยาวขึ้นได้
        temperature=0.3,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.1,
        return_full_text=False,
        pad_token_id=tokenizer.eos_token_id # แก้ปัญหา Warning สแปมหน้าจอ
    )
    
    return HuggingFacePipeline(pipeline=pipe)

# --- 3. ปรับจูน Prompt Engineering สำหรับแนะนำหนังและภาษาไทย ---
prompt_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a knowledgeable and friendly movie expert assistant. Your job is to answer the user's question using the provided movie plot context.

CRITICAL RULES:
1. If the user asks for a movie recommendation based on a genre or theme (e.g., "หนังผี", "action", "ผีหลอกๆ"), analyze the context and recommend the movies that fit the description. Do NOT say you don't know the movie name.
2. ALWAYS answer in the SAME LANGUAGE as the user's question. If the question is in Thai, translate your thoughts and answer in Thai naturally.
3. If the exact answer is not in the context, you can use your general knowledge, but prioritize the context first.
<|eot_id|><|start_header_id|>user<|end_header_id|>

Here is the movie plot context:
{context}

Based on the context above, answer the following question:
{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

PROMPT = PromptTemplate(
    template=prompt_template, 
    input_variables=["context", "question"]
)

def answer_question(question, vectorstore, llm):
    """ค้นหาข้อมูลและส่งให้ LLM ตอบคำถาม"""
    docs = vectorstore.invoke(question)
    
    contexts = []
    sources = []
    
    for doc in docs:
        title = doc.metadata.get("title", "Unknown")
        genre = doc.metadata.get("genre", "Unknown")
        year = doc.metadata.get("year", "Unknown")
        director = doc.metadata.get("director", "Unknown")
        cast = doc.metadata.get("cast", "Unknown")
        
        context_str = (
            f"Title: {title}\n"
            f"Release Year: {year}\n"
            f"Director: {director}\n"
            f"Cast: {cast}\n"
            f"Genre: {genre}\n"
            f"Plot: {doc.page_content}"
        )
        contexts.append(context_str)
        
        sources.append({
            "title": title,
            "genre": genre,
            "year": year,
            "director": director,
            "cast": cast,
            "score": "Hybrid",
            "content": doc.page_content
        })
        
    combined_context = "\n\n".join(contexts)
    final_prompt = PROMPT.format(context=combined_context, question=question)
    
    print(f"\nGenerating answer for: '{question}'...")
    
    # ส่ง Stop words ให้ LLM หยุดพิมพ์เมื่อจบประโยค
    answer = llm.invoke(final_prompt, stop=["<|eot_id|>", "<|end_of_text|>"])
    
    return {
        "answer": answer.strip(),
        "sources": sources
    }

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    index_dir = os.path.join(base_dir, "data", "faiss_index")
    
    if not os.path.exists(index_dir):
        print(f"Error: Vector index not found at {index_dir}. Run build_index.py first.")
        sys.exit(1)
        
    print("Loading resources...")
    vectorstore = load_vectorstore(index_dir)
    llm = load_llm()
    
    # ทดสอบด้วยคำถามภาษาไทยและแนวหนัง
    test_q = "แนะนำหนังแนวผีหลอกๆ ให้หน่อยครับ"
    result = answer_question(test_q, vectorstore, llm)
    
    print("\n" + "="*50)
    print(f"Question: {test_q}")
    print(f"Answer: {result['answer']}")
    print("="*50)
    print("\nSources:")
    for i, src in enumerate(result['sources']):
        print(f"{i+1}. Movie: {src['title']} ({src['year']}) - Genre: {src['genre']}")