from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from transformers import pipeline
import pandas as pd

# Step 1: Load the Medical Dataset
data = pd.read_csv("medical_knowledge.csv")  # Replace with your dataset file
data_dict = data.to_dict('records')  # Convert to a dictionary for easier handling

# Step 2: Create Embeddings for the Dataset
encoder = SentenceTransformer('all-MiniLM-L6-v2')  # A lightweight transformer model for embeddings
embeddings = [encoder.encode(record["content"]).tolist() for record in data_dict]

# Step 3: Initialize and Set Up the Vector Database (Qdrant)
qdrant = QdrantClient(":memory:")  # In-memory vector DB for this example
qdrant.recreate_collection(
    collection_name="medical_base",
    vectors_config=models.VectorParams(
        size=encoder.get_sentence_embedding_dimension(),  # Embedding size
        distance=models.Distance.COSINE  # Similarity metric
    )
)

# Step 4: Push Medical Knowledge into the Vector Database
records = [
    models.Record(id=i, vector=embeddings[i], payload=data_dict[i])
    for i in range(len(data_dict))
]
qdrant.upload_records(collection_name="medical_base", records=records)

# Step 5: Define the RAG Loop
def medical_rag_loop(query, vector_db, llm):
    # Encode the query into a vector
    query_vector = encoder.encode(query).tolist()
    
    # Retrieve relevant records from the vector database
    search_results = vector_db.search(
        collection_name="medical_base",
        query_vector=query_vector,
        limit=3  # Top 3 matches
    )
    
    # Combine retrieved context
    retrieved_context = "\n".join([hit.payload["content"] for hit in search_results])
    print("Retrieved Context:\n", retrieved_context)

    # Generate a response using the LLM
    response = llm(f"{retrieved_context}\n\nAnswer the query: {query}")
    return response

# Step 6: Load a Local LLM for Medical Q&A or use API to use any desired LLM model. Here I simply used gpt2
llm_model = pipeline("text-generation", model="gpt2", tokenizer="gpt2")

# Example Query
query = "What are the symptoms of diabetes?"
response = medical_rag_loop(query, qdrant, llm_model)
print("LLM Response:\n", response)
