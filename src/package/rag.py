import os 
import faiss
import cohere
import numpy as np
from data.news import documents
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()

client = OpenAI()
co = cohere.Client()

EMBEDDINGS_PATH = "embeddings.npy"
INDEX_PATH = "faiss.index"

# EMBDED AND INDEX
def get_embeddings(texts):
    """Convert the documents to embeddings"""
    response = client.embeddings.create(
        input=texts,
        model="text-embedding-3-small"
    )
    return [item.embedding for item in response.data]

def load_or_build_index():
    if os.path.exists(EMBEDDINGS_PATH) and os.path.exists(INDEX_PATH):
        print("Loading existing index from disk...")
        embeddings = np.load(EMBEDDINGS_PATH)
        index = faiss.read_index(INDEX_PATH)
        print(f"Loaded {index.ntotal} chunks from disk\n")
    else:
        print("Building index for the first time...")
        embeddings = np.array(get_embeddings(documents)).astype("float32")
        index = faiss.IndexFlatL2(1536)
        index.add(embeddings)
        np.save(EMBEDDINGS_PATH, embeddings)
        faiss.write_index(index, INDEX_PATH)
        print(f" Indexed and saved {index.ntotal} chunks\n")
    return index

index = load_or_build_index()



# RETRIEVE
def retrieve(query, k=10):
    "Vanilla retrieve on the embeddings"
    query_embedding = np.array(get_embeddings([query])).astype("float32")
    # Search FAISS for top-k most similar chunks
    distances, indices = index.search(query_embedding, k)
    # Use the indices to look up the actual text
    results = [(documents[i], distances[0][j]) for j, i in enumerate(indices[0])]
    return results

#RERANK
def rerank(query, retrieved_chunks, top_n=5):
    # Extract just the text from retrived chunks
    docs = [chunk for chunk, score in retrieved_chunks]
    # Cohere scores each pair
    response = co.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=docs,
        top_n=top_n
    )
   
    # Reranked chunks
    return [(docs[r.index], r.relevance_score) for r in response.results]
    
# LLM SYNTHESIS
def synthesise_risk_summary(query, retrieved):
    # Format the chunks into context
    context = "\n\n".join([f"- {chunk}" for chunk, score in retrieved])    
    # Prompt
    prompt = f"""You are a financial crime compliance analyst.
    
    Based on the following adverse media excerpts, write a concise risk summary for the entity: {query}

    The summary should cover:
    - Nature of the adverse findings
    - Regulatory or legal actions taken
    - Financial penalties if any
    - Overall risk rating (High / Medium / Low)

    Adverse Media Excerpts:
    {context}

    Risk Summary:"""

    # LLM
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a financial crime compliance analyst specialising in adverse media screening."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2  # low temperature for factual consistent output
    )
   
    return response.choices[0].message.content

# PIPELINE
def run_pipeline(query):
    print("=" * 60)
    print(f"QUERY: {query}")
    print("=" * 60)

    # VANILLA RETRIEVAL
    retrieved = retrieve(query, k=10)

    print("\n── VANILLA FAISS RETRIEVAL (top 5) ──")
    for i, (chunk, score) in enumerate(retrieved[:5]):
        print(f"{i+1}. [L2: {score:.4f}] {chunk[:120]}...")

    # VANILLA SYNTHESIS
    vanilla_summary = synthesise_risk_summary(query, retrieved[:5])
    print("\n── VANILLA RISK SUMMARY ──")
    print(vanilla_summary)

    # RERANKED RETRIEVAL
    reranked = rerank(query, retrieved, top_n=5)

    print("\n── RERANKED RETRIEVAL (top 5) ──")
    for i, (chunk, score) in enumerate(reranked):
        print(f"{i+1}. [Relevance: {score:.4f}] {chunk[:120]}...")

    # RERANKED SYNTHESIS
    reranked_summary = synthesise_risk_summary(query, reranked)
    print("\n── RERANKED RISK SUMMARY ──")
    print(reranked_summary)

    print("\n" + "=" * 60)

# ENTRY POINT 
if __name__ == "__main__":
    run_pipeline("Danske Bank")


