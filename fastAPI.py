import os
import google.generativeai as genai
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import re
from typing import List, Dict, Any
from collections import deque
from fastapi import FastAPI, HTTPException, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import uuid

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Harry Potter Bot API",
    description="An API for conversing about the Harry Potter universe",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Pinecone client
try:
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
except Exception as e:
    print(f"Error initializing Pinecone: {str(e)}")

# Define Serverless specifications
serverless_spec = ServerlessSpec(
    cloud="aws",
    region="us-east-1"
)

# Index name
index_name = "harry-potter-and-the-sorcerers-stone"

# Try to create index if it doesn't exist
try:
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=768,
            metric="cosine",
            spec=serverless_spec
        )
    
    # Connect to the index
    index = pc.Index(index_name)
except Exception as e:
    print(f"Error with Pinecone index: {str(e)}")

# Initialize Gemini
try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
except Exception as e:
    print(f"Error initializing Gemini: {str(e)}")

# Dictionary to store conversation sessions
conversation_sessions = {}

# Conversation memory class
class ConversationMemory:
    def __init__(self, max_history=5):
        self.history = deque(maxlen=max_history)
        self.session_id = str(uuid.uuid4())
        
    def add_interaction(self, user_message, bot_response):
        self.history.append({"user": user_message, "bot": bot_response})
        
    def get_conversation_context(self):
        context = ""
        if self.history:
            context = "Previous conversation:\n"
            for i, interaction in enumerate(self.history):
                context += f"User: {interaction['user']}\n"
                context += f"Bot: {interaction['bot']}\n\n"
        return context
    
    def clear(self):
        self.history.clear()

# API request and response models
class ChatRequest(BaseModel):
    message: str
    session_id: str = None

class ChatResponse(BaseModel):
    response: str
    session_id: str

# Function to embed text using Gemini's `text-embedding-004` model
def embed_text_with_gemini(text: str) -> List[float]:
    if text.strip():
        try:
            response = genai.embed_content(
                model='text-embedding-004',
                content=text,
                task_type="retrieval_document"
            )
            return response['embedding']
        except Exception as e:
            print(f"Error embedding text: {str(e)}")
            return None
    else:
        print("Warning: Skipping empty text.")
        return None

# Function to rerank retrieved passages by relevance
def rerank_passages(passages: List[str], query: str) -> List[str]:
    """Rerank passages based on relevance to query"""
    query_words = set(query.lower().split())
    
    # Calculate relevance score for each passage
    passage_scores = []
    for passage in passages:
        # Count keyword matches
        keyword_matches = sum(1 for word in query_words if word in passage.lower())
        # Prioritize passages with more query words
        passage_scores.append((passage, keyword_matches))
    
    # Sort passages by score (descending)
    sorted_passages = [p for p, _ in sorted(passage_scores, key=lambda x: x[1], reverse=True)]
    return sorted_passages

# Function to retrieve the most relevant context for a query
def retrieve_relevant_documents(query: str, top_k: int = 15) -> List[str]:
    """Retrieve and rank relevant documents from Pinecone"""
    query_embedding = embed_text_with_gemini(query)
    
    if not query_embedding:
        return []

    try:
        # Query Pinecone
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )

        # Extract relevant texts with their scores
        relevant_texts = [match['metadata'].get('text', '') for match in results['matches']]
        
        # Rerank passages based on query relevance
        reranked_passages = rerank_passages(relevant_texts, query)
        
        return reranked_passages
    except Exception as e:
        print(f"Error retrieving documents: {str(e)}")
        return []

# Function to remove duplicate or highly similar content
def deduplicate_content(passages: List[str]) -> List[str]:
    """Remove duplicate or highly similar passages"""
    unique_passages = []
    for passage in passages:
        # Normalize for comparison
        normalized = re.sub(r'\s+', ' ', passage.lower().strip())
        
        # Check if this passage is similar to any we've already kept
        is_duplicate = False
        for existing in unique_passages:
            existing_norm = re.sub(r'\s+', ' ', existing.lower().strip())
            
            # Simple similarity check - could be improved with more sophisticated methods
            # Consider duplicate if 80% of words match
            if len(set(normalized.split()) & set(existing_norm.split())) / len(set(normalized.split() + existing_norm.split())) > 0.8:
                is_duplicate = True
                break
                
        if not is_duplicate:
            unique_passages.append(passage)
            
    return unique_passages

# Function to combine passages into coherent context
def build_context(passages: List[str], max_length: int = 6000) -> str:
    """Build a coherent context from the retrieved passages"""
    # Remove duplicates
    unique_passages = deduplicate_content(passages)
    
    # Join passages and truncate if necessary
    context = " ".join(unique_passages)
    if len(context) > max_length:
        context = context[:max_length]
    
    return context

# Improved prompt engineering for conversational answers
def generate_conversational_response(context: str, user_message: str, conversation_history: str) -> str:
    try:
        # Initialize the model
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Conversational prompt that maintains Harry Potter character and includes history
        prompt = f"""You are a friendly, conversational expert on the Harry Potter universe, especially 'Harry Potter and the Sorcerer's Stone'. 
        Respond to the user's message in a natural, engaging way as if you're having a conversation about the wizarding world.

        {conversation_history}

        Book context relevant to the current message:
        {context}

        User's current message: "{user_message}"

        Guidelines for your response:
        1. Respond conversationally as if you're chatting with a fellow Harry Potter fan
        2. Seamlessly blend information from the book with your knowledge of Harry Potter
        3. Stay in character as a friendly wizard/witch who knows the Harry Potter universe well
        4. Reference previous conversation points when relevant
        5. Keep your response concise but informative
        6. Use occasional wizarding expressions to maintain authenticity
        7. If you don't know something, be honest but stay in character
        8. Make sure each response is unique - avoid reusing the same phrases or clichés
        9. Use proper markdown formatting for emphasis, headings, and lists where appropriate
        10. Vary your sentence structure and vocabulary to keep the conversation fresh

        Your conversational response:"""
        
        # Generate the response with settings for natural conversation
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,  # Higher temperature for more conversational feel
                top_p=0.95,
                top_k=40,
                max_output_tokens=800
            )
        )
        
        return response.text.strip()
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        return "*Sorry, I seem to have been hit by a Confundus Charm!* Could you ask me again?\n\n*(Error occurred while generating response)*"

def chat_with_harry_potter_bot(user_message: str, session_id: str = None) -> tuple[str, str]:
    """Process a user message and generate a conversational response"""
    # Handle session
    if session_id and session_id in conversation_sessions:
        memory = conversation_sessions[session_id]
    else:
        memory = ConversationMemory(max_history=5)
        session_id = memory.session_id
        conversation_sessions[session_id] = memory
    
    # Get conversation history
    conversation_context = memory.get_conversation_context()
    
    # Special commands
    if user_message.lower() == "clear history":
        memory.clear()
        return "Obliviate! Our conversation history has been cleared.", session_id
    
    # 1. Retrieve relevant passages from the book
    passages = retrieve_relevant_documents(user_message, top_k=20)
    
    if not passages:
        # No relevant passages found, but still try to answer based on general knowledge
        context = "No specific book passages found."
    else:
        # 2. Build optimized context
        context = build_context(passages)
    
    # 3. Generate conversational response
    response = generate_conversational_response(context, user_message, conversation_context)
    
    # 4. Update conversation memory
    memory.add_interaction(user_message, response)
    
    return response, session_id

# API endpoints
@app.get("/")
def read_root():
    return {"message": "Welcome to the Harry Potter Bot API! Send a POST request to /chat to start a conversation."}

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(chat_request: ChatRequest):
    response, session_id = chat_with_harry_potter_bot(
        chat_request.message, 
        chat_request.session_id
    )
    return ChatResponse(response=response, session_id=session_id)

@app.post("/clear-history")
def clear_history(request: Dict[str, str] = Body(...)):
    session_id = request.get("session_id")
    if session_id and session_id in conversation_sessions:
        conversation_sessions[session_id].clear()
        return {"message": "Conversation history cleared", "session_id": session_id}
    else:
        raise HTTPException(status_code=404, detail="Session not found")

# Session cleanup (could be improved with scheduled cleanup for inactive sessions)
# In a production environment, consider adding a background task to clean up old sessions

# Run the server if executed directly
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)