import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
vector_size = 384

qdrant_client = QdrantClient(url="http://localhost:6333")

collection_name = "gym_exercises"

print(f"Checking for existing collection '{collection_name}'")
if qdrant_client.collection_exists(collection_name=collection_name):
    print("Collection exists. Checking current document count...")
    collection_info = qdrant_client.get_collection(collection_name=collection_name)
    current_count = collection_info.points_count
    print(f"Current collection has {current_count} documents.")
    
    if current_count > 0:
        print("Collection already has data. Skipping ingestion to avoid duplicates.")
        print("If you want to refresh the data, manually delete the collection first.")
        exit(0)
    else:
        print("Collection exists but is empty. Will add new data.")
else:
    print(f"Creating new collection: '{collection_name}'")
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE,
        ),
    )


df = pd.read_csv("data/megaGymDataset.csv")
df = df.dropna(subset=['Title', 'Desc', 'Type', 'BodyPart', 'Equipment', 'Level'])
df = df.astype(str)
df['id'] = df.index

def create_document_text(row):
    return (
        f"Title: {row['Title']}\n"
        f"Description: {row['Desc']}\n"
        f"Type: {row['Type']}\n"
        f"Body Part: {row['BodyPart']}\n"
        f"Equipment: {row['Equipment']}\n"
        f"Level: {row['Level']}"
    )

df["document_text"] = df.apply(create_document_text, axis=1)

print(f"Loaded and prepared {len(df)} documents for ingestion.")

batch_size = 100
for i in tqdm(range(0, len(df), batch_size)):
    batch_df = df.iloc[i : i + batch_size]
    texts_to_embed = batch_df["document_text"].tolist()

    embeddings = embedding_model.encode(texts_to_embed)
    
    vectors = embeddings.tolist()

    qdrant_client.upsert(
        collection_name=collection_name,
        points=models.Batch(
            ids=batch_df["id"].tolist(),
            vectors=vectors,
            payloads=[
                {
                    "title": row["Title"],
                    "type": row["Type"],
                    "muscle": row["BodyPart"],
                    "equipment": row["Equipment"],
                    "level": row["Level"],
                    "text": row["document_text"],
                }
                for _, row in batch_df.iterrows()
            ],
        ),
        wait=False
    )

print("Your data has been successfully ingested into Qdrant")