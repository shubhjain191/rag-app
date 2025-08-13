import os
import openai
import instructor
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional, List
from sentence_transformers import SentenceTransformer

load_dotenv()

OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found in .env file")

openrouter_client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    default_headers={
        "HTTP-Referer": "http://localhost:3000",
        "X-Title": "RAG App",
    },
)

instructor_client = instructor.patch(openrouter_client)

qdrant_client = QdrantClient(url="http://localhost:6333")

collection_name = "gym_exercises"
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
generation_model_name = "openai/gpt-3.5-turbo" 


class QdrantFilter(BaseModel):
    level: Optional[str] = Field(None, description="The difficulty level of the exercise, e.g., 'Beginner', 'Intermediate', 'Expert'")
    muscle: Optional[str] = Field(None, description="The primary muscle group targeted, e.g., 'Chest', 'Legs', 'Back'")
    equipment: Optional[str] = Field(None, description="The equipment required, e.g., 'Dumbbells', 'Barbell', 'Body Only'")

class SearchQuery(BaseModel):
    query: str = Field(..., description="The user's query, transformed into a clear, semantic search query for a vector database.")
    filters: List[QdrantFilter] = Field(..., description="A list of structured filters extracted from the user's query.")


def get_structured_query(user_query: str) -> SearchQuery:
    return instructor_client.chat.completions.create(
        model=generation_model_name,
        response_model=SearchQuery,
        messages=[
            {"role": "system", "content": "You are a world-class query understanding engine. Your task is to convert a user's natural language query into a structured object containing a semantic search query and a list of key-value filters for a Qdrant database."},
            {"role": "user", "content": f"Analyze and structure the following query: '{user_query}'"},
        ],
    )

def build_qdrant_filter(structured_query: SearchQuery) -> Optional[models.Filter]:
    if not structured_query.filters:
        return None
    
    filter_conditions = []
    
    for f in structured_query.filters:
        if f.level:
            filter_conditions.append(models.FieldCondition(key="level", match=models.MatchValue(value=f.level)))
        if f.muscle:
            filter_conditions.append(models.FieldCondition(key="muscle", match=models.MatchValue(value=f.muscle)))
        if f.equipment:
            filter_conditions.append(models.FieldCondition(key="equipment", match=models.MatchValue(value=f.equipment)))
            
    if not filter_conditions:
        return None
        
    return models.Filter(must=filter_conditions)

def run_query(user_query: str):
    print(f"Processing query: '{user_query}'")
    
    structured_query = get_structured_query(user_query)
    print(f"-> Understood query: {structured_query.query}")
    print(f"-> Applying filters: {structured_query.filters}")

    query_vector = embedding_model.encode(structured_query.query).tolist()
    
    query_filter = build_qdrant_filter(structured_query)

    search_results = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        query_filter=query_filter,
        limit=3
    )
    
    prompt_instructions = "You are an expert fitness AI. Based *only* on the provided context, write a clear and concise answer to the user's question."
    
    context_strings = [result.payload['text'] for result in search_results]
    context_block = "\n---\n".join(context_strings)

    final_prompt = f"""
    {prompt_instructions}

    CONTEXT:
    {context_block}

    USER'S QUESTION:
    {user_query}
    """

    response = openrouter_client.chat.completions.create(
        model=generation_model_name,
        messages=[{"role": "user", "content": final_prompt}],
    )
    
    print("Response:")
    print(response.choices[0].message.content)

if __name__ == "__main__":
    user_questions = [
        "Show me some beginner friendly leg exercises using dumbbells",
        "What are the best back exercises for an expert without equipment?",
        "chest workout using a barbell",
    ]
    for q in user_questions:
        run_query(q)