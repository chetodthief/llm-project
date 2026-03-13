import os
import pandas as pd
import json
import re
from langchain_text_splitters import TokenTextSplitter

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def process_and_chunk_data(input_csv, output_json):
    print(f"Loading data from {input_csv}...")
    # df = pd.read_csv(input_csv)
    df = pd.read_csv(input_csv, nrows=1000)
    
    # For the new dataset, the text column is 'Plot'
    df['Plot'] = df['Plot'].apply(clean_text)
    
    # Drop rows with empty plots
    df = df[df['Plot'] != ""]
    
    # Initialize TokenTextSplitter for 300 token chunks
    text_splitter = TokenTextSplitter(chunk_size=300, chunk_overlap=30)
    
    chunks = []
    chunk_id_counter = 0
    
    for _, row in df.iterrows():
        title = row.get('Title', 'Unknown')
        genre = row.get('Genre', 'Unknown')
        year = row.get('Release Year', 'Unknown')
        origin = row.get('Origin/Ethnicity', 'Unknown')
        director = row.get('Director', 'Unknown')
        cast = row.get('Cast', 'Unknown')
        wiki = row.get('Wiki Page', 'Unknown')
        plot = row.get('Plot', '')
        
        # Split plot into chunks
        splits = text_splitter.split_text(str(plot))
        for split in splits:
            chunks.append({
                "chunk_id": chunk_id_counter,
                "text": split,
                "metadata": {
                    "title": str(title),
                    "genre": str(genre),
                    "year": str(year),
                    "origin": str(origin),
                    "director": str(director),
                    "cast": str(cast),
                    "wiki_page": str(wiki)
                }
            })
            chunk_id_counter += 1
            
    # Save the chunked data
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=4)
        
    print(f"Data processing complete. Saved {len(chunks)} chunks to {output_json}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_file = r"D:\llm\llm-project\wiki_movie_plots_deduped.csv\wiki_movie_plots_deduped.csv"

    output_file = os.path.join(base_dir, "data", "chunked_plots.json")
    process_and_chunk_data(input_file, output_file)
