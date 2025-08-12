import os
import pandas as pd
import google.generativeai as genai
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")

# qdrant client
print("Initializing Qdrant client...")
qdrant_client = QdrantClient(url="http://localhost:6333")

# gemini client
print("Initializing Gemini client...")
genai.configure(api_key=GEMINI_API_KEY)

# create collection
collection_name = "gym_exercises"
print(f"Creating Qdrant collection: '{collection_name}'")
try:
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=768,
            distance=models.Distance.COSINE
        ),
    )
    print("Collection created successfully.")
except Exception as e:
    print(f"Collection may already exist... Info: {e}")


# load data
print("Loading data from CSV file...")
df = pd.read_csv("data/megaGymDataset.csv")

df['id'] = df.index

# create document text
def create_document_text(row):
    return (
        f"Title: {row['Title']}\n"
        f"Description: {row['Desc']}\n"
        f"Type: {row['Type']}\n"
        f"Body Part: {row['BodyPart']}\n"
        f"Equipment: {row['Equipment']}\n"
        f"Level: {row['Level']}\n"
        f"Rating: {row['Rating']}\n"
        f"Rating Description: {row['RatingDesc']}"
    )

df['document_text'] = df.apply(create_document_text, axis=1)
print(f"Loaded and prepared {len(df)} documents.")


print("Starting ingestion..")

batch_size = 32
for i in tqdm(range(0, len(df), batch_size)):
    batch_df = df.iloc[i:i+batch_size]

    texts_to_embed = batch_df['document_text'].tolist()

    #gemini api
    embedding_response = genai.embed_content(
        model='models/embedding-001',
        content=texts_to_embed,
        task_type="RETRIEVAL_DOCUMENT"
    )
    vectors = embedding_response['embedding']

    # qdrant api
    qdrant_client.upsert(
        collection_name=collection_name,
        points=models.Batch(
            ids=batch_df['id'].tolist(),
            vectors=vectors,
            payloads=[
                {
                    "title": row["Title"],
                    "type": row["Type"],
                    "muscle": row["BodyPart"],
                    "equipment": row["Equipment"],
                    "text": row["document_text"],
                } for index, row in batch_df.iterrows()
            ]
        ),
        wait=True
    )

print("Your data has been successfully ingested into Qdrant")