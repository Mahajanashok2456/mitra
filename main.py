from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import pymongo
from pymongo import MongoClient
import google.generativeai as genai
from dotenv import load_dotenv
import json
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Mahabharata & Ramayana Chat API", version="1.0.0")

# Add CORS middleware
# Allow both local development and production frontend URLs
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class ChatRequest(BaseModel):
    question: str
    history: Optional[List[Dict[str, str]]] = []

class TestRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str
    intent: str
    source_evidence: List[Dict[str, Any]]

# Global variables for MongoDB and Gemini
mongodb_client: Optional[MongoClient] = None
gemini_model: Optional[genai.GenerativeModel] = None
EMBEDDING_MODEL_NAME = "models/text-embedding-004"

def init_mongodb():
    """Initialize MongoDB connection"""
    global mongodb_client
    try:
        atlas_uri = os.getenv("ATLAS_URI")
        if not atlas_uri:
            raise ValueError("ATLAS_URI environment variable not set")

        mongodb_client = MongoClient(atlas_uri)
        # Test the connection
        mongodb_client.admin.command('ping')
        logger.info("Connected to MongoDB successfully")
        return mongodb_client
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise

def init_gemini():
    """Initialize Gemini AI"""
    global gemini_model
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")

        genai.configure(api_key=api_key)

        # Use the specific model requested
        try:
            gemini_model = genai.GenerativeModel('gemini-2.5-flash')
            logger.info("Gemini AI initialized successfully with gemini-2.5-flash")
            return gemini_model
        except Exception as e:
            logger.error(f"Failed to initialize Gemini AI with gemini-2.5-flash: {e}")
            raise Exception(f"Failed to initialize Gemini AI with gemini-2.5-flash. Please check your API key and internet connection. Error: {e}")
    except Exception as e:
        logger.error(f"Failed to initialize Gemini AI: {e}")
        raise

def router_function(message: str) -> str:
    """
    Classify user intent as factual, guidance, or general
    """
    try:
        if gemini_model is None:
            logger.error("Gemini model not initialized")
            return "general"

        router_prompt = f"""
        Classify the user message into EXACTLY ONE category.

        Categories:
        - factual → objective questions about Mahabharata or Ramayana (people, events, places, definitions)
        - guidance → emotional pain, confusion, moral conflict, life advice, inner struggle
        - general → greetings, casual talk, unrelated questions

        Rules:
        - Return ONLY ONE lowercase word.
        - No explanations.
        - No punctuation.
        - No extra text.
        - If unsure, return "general".

        User message:
        "{message}"
        """

        response = gemini_model.generate_content(router_prompt)
        raw_intent = response.text.strip().lower()
        intent = ''.join(c for c in raw_intent if c.isalnum())

        # Ensure we return only valid intents
        if intent not in ["factual", "guidance", "general"]:
            intent = "general"

        return intent
    except Exception as e:
        logger.error(f"Error in router function: {e}")
        return "general"

def get_adaptive_prompt(question: str, context: List[Dict], history: Optional[List[Dict[str, str]]] = None, intent: str = "guidance") -> str:
    """
    Generates a prompt that:
    1. Analyzes user tone/intent.
    2. Uses retrieved context (Knowledge Base).
    3. Frames the answer accordingly.
    4. RECOMMENDS responses based on conversation history.
    """
    context_str = "\n".join([f"Info: {doc.get('text', '')}" for doc in context[:4]])

    if history is None:
        history = []

    # Format history for the prompt
    history_str = ""
    if history:
        history_str = "\n".join(
            [f"{msg['role'].upper()}: {msg['content']}" for msg in history[-2:]] # Cost opt: Keep last 2 turns
        )

    # FACTUAL INTENT - EXPERT PERSONA
    if intent == "factual":
        return f"""Answer the question with factual accuracy.

Rules:
- 2–3 sentences ONLY.
- Stop immediately after the answer.
- Do not add background, interpretation, or philosophy.

User question:
"{question}"

Context:
{context_str}
"""

    # GUIDANCE INTENT (DEFAULT) - THERAPIST PERSONA
    return f"""Respond with emotional clarity and restraint.

Rules:
- Maximum 4–5 sentences.
- No repetition.
- No lecturing.
- Emotional safety > sounding wise.

If there is any conflict between:
- speed and safety
- verbosity and clarity
- wisdom and emotional grounding

Always choose:
- safety
- clarity
- restraint

Previous conversation:
{history_str}

User input:
"{question}"

Context:
{context_str}
"""

def perform_vector_search(query: str, limit: int = 5) -> List[Dict]:
    """Perform vector search on MongoDB"""
    try:
        if mongodb_client is None:
            logger.error("MongoDB client not initialized")
            return []

        db = mongodb_client["mahabharata_db"]
        collection = db["texts"]

        # Generate embedding for the query using Gemini
        result = genai.embed_content(
            model=EMBEDDING_MODEL_NAME,
            content=query,
            task_type="retrieval_query"
        )

        query_embedding = result['embedding']

        # Perform vector search
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "text_embedding",
                    "queryVector": query_embedding,
                    "numCandidates": limit * 2,
                    "limit": limit
                }
            },
            {
                "$project": {
                    "text": 1,
                    "source": 1,
                    "chapter": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]

        results = list(collection.aggregate(pipeline))
        return results
    except Exception as e:
        logger.error(f"Error in vector search: {e}")
        return []

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint"""
    try:
        logger.info(f"Received question: {request.question}")

        # Check if services are initialized
        if mongodb_client is None:
            logger.error("MongoDB client is None")
            return ChatResponse(
                answer="MongoDB connection failed. Please check your ATLAS_URI in the .env file.",
                intent="general",
                source_evidence=[]
            )

        if gemini_model is None:
            logger.error("Gemini model is None")
            return ChatResponse(
                answer="Gemini AI initialization failed. Please check your GEMINI_API_KEY in the .env file.",
                intent="general",
                source_evidence=[]
            )

        # CRISIS DETECTION
        crisis_keywords = [
            "suicide", "kill myself", "want to die", "end my life",
            "hurt myself", "no reason to live", "better off dead",
            "ending it all", "don't want to live"
        ]
        
        if any(keyword in request.question.lower() for keyword in crisis_keywords):
            logger.warning("Crisis keywords detected. Bypassing LLM.")
            return ChatResponse(
                answer="""I hear how heavy your heart is right now, and I want you to know that you are not alone in this pain.

Please reach out to a human voice who can truly support you:
📞 India: 9820466726 (AASRA) or 14416 (KIRAN Mental Health Helpline)

Your life has value, even in this dark moment. Please call them.""",
                intent="guidance",
                source_evidence=[]
            )

        # Classify intent
        intent = router_function(request.question)
        logger.info(f"Classified intent: {intent}")

        response_data = {
            "answer": "",
            "intent": intent,
            "source_evidence": []
        }

        if intent in ["factual", "guidance"]:
            expanded_query = request.question
            
            # COST OPT: Only expand query for guidance, factual usually doesn't need it
            if intent == "guidance":
                try:
                    expansion_prompt = f"""
                    Extract 2–3 short keywords or phrases that capture the CORE theme of the user’s question,
                    focused on emotional states, moral dilemmas, or epic concepts from Ramayanam or Mahabharat.

                    Rules:
                    - Output ONLY comma-separated keywords.
                    - No sentences.
                    - No explanations.
                    - If nothing fits, return an empty response.

                    User question:
                    "{request.question}"

                    Examples:
                    betrayal, dharma dilemma, grief
                    identity conflict, exile, resilience
                    """
                    expansion_response = gemini_model.generate_content(expansion_prompt)
                    themes = [theme.strip() for theme in expansion_response.text.strip().split(',') if theme.strip()]
                    if themes:
                        expanded_query = request.question + " " + " ".join(themes)
                        logger.info(f"Expanded query: {expanded_query}")
                except Exception as e:
                    logger.warning(f"Query expansion failed: {e}. Using original question.")

            # COST OPT: Skip vector search for short factual questions to save latency
            if intent == "factual" and len(request.question) < 60:
                logger.info("Skipping vector search for short query")
                search_results = []
            else:
                # Perform vector search for context using expanded query
                search_results = perform_vector_search(expanded_query)

            if search_results:
                response_data["source_evidence"] = [
                    {
                        "text": doc.get("text", ""),
                        "source": doc.get("source", "Unknown"),
                        "chapter": doc.get("chapter", ""),
                        "score": doc.get("score", 0)
                    }
                    for doc in search_results
                ]

                # Use the new unified adaptive prompt for both factual and guidance
                prompt = get_adaptive_prompt(request.question, search_results, request.history, intent)

                # Get response from Gemini
                # Cost optimization: enforce max tokens
                response = gemini_model.generate_content(prompt, generation_config={"max_output_tokens": 250})
                response_data["answer"] = response.text
            else:
                # Use the same unified prompt but with empty context (triggers fallback logic inside)
                prompt = get_adaptive_prompt(request.question, [], request.history, intent)
                response = gemini_model.generate_content(prompt)
                response_data["answer"] = response.text
        else:
            # General conversation
            general_prompt = f"""
            Respond naturally and briefly.

            Rules:
            - Friendly and human.
            - No advice unless asked.
            - No therapy tone.
            - No epic references.

            User message:
            "{request.question}"
            """

            response = gemini_model.generate_content(general_prompt)
            response_data["answer"] = response.text

        return ChatResponse(**response_data)

    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return ChatResponse(
            answer=f"An error occurred: {str(e)}. Please check the backend logs for details.",
            intent="general",
            source_evidence=[]
        )

@app.on_event("startup")
async def startup_event():
    """Initialize connections on startup"""
    try:
        init_mongodb()
        init_gemini()
        logger.info("Application startup complete")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        # Don't raise the exception - let the server start anyway
        # This way the /chat endpoint can still return meaningful errors

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Mahabharata & Ramayana Chat API", "status": "running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    mongo_status = "connected" if mongodb_client is not None else "disconnected"
    gemini_status = "initialized" if gemini_model is not None else "failed"
    status = "healthy" if mongo_status == "connected" and gemini_status == "initialized" else "unhealthy"
    return {"status": status, "mongodb": mongo_status, "gemini": gemini_status}

@app.post("/test")
async def simple_test_endpoint(request: TestRequest):
    """Simple test endpoint that doesn't use MongoDB or Gemini"""
    return {
        "message": "Test successful!",
        "received_question": request.question,
        "length": len(request.question)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)