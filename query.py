import os
import google.generativeai as genai
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")

# gemini and qdrant client
qdrant_client = QdrantClient(url="http://localhost:6333")
genai.configure(api_key=GEMINI_API_KEY)

# collection name
collection_name = "gym_exercises"

# list of questions
user_questions = [
    "What's a good way to workout my chest?",
    "Show me some beginner friendly leg exercises using dumbbells",
    "What are the best back exercises for intermediate level?",
    "Can you recommend cardio workouts that don't require any gym equipments?",
]

# main
for user_question in user_questions:

    print(f"Searching for: '{user_question}'")

    # embedding
    embedding_response = genai.embed_content(
        model='models/embedding-001',
        content=user_question,
        task_type="RETRIEVAL_QUERY"
    )
    query_vector = embedding_response['embedding']

    # search qdrant
    search_results = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=3
    ).points

    # prompt
    prompt_instructions = """
    You are an expert fitness AI. A user has asked a question.
    Using the following provided context, which contains relevant exercises from your knowledge base,
    please write a clear, helpful, and concise answer.
    Directly answer the user's question. Do not say "Based on the context...".
    """

    context_strings = []
    for result in search_results:
        payload = result.payload
        context_strings.append(
            f"Title: {payload['title']}\n"
            f"Description: {payload['text']}\n---\n"
        )
    context_block = "\n".join(context_strings)

    final_prompt = f"""
    {prompt_instructions}

    CONTEXT:
    {context_block}

    USER'S QUESTION:
    {user_question} 
    """

    # generate response
    generative_model = genai.GenerativeModel('gemini-2.0-flash')
    response = generative_model.generate_content(final_prompt)
    
    print("Response:")
    print(response.text)