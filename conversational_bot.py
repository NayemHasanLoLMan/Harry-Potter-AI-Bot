import os
import google.generativeai as genai
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import re
from typing import List, Dict, Any
from collections import deque

# Load environment variables from .env file
load_dotenv()

# Initialize Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Define Serverless specifications
serverless_spec = ServerlessSpec(
    cloud="aws",
    region="us-east-1"
)

# Index name
index_name = "harry-potter-and-the-sorcerers-stone"

# Create index if it doesn't exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=serverless_spec
    )

# Connect to the index
index = pc.Index(index_name)

# Initialize Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Conversation memory to store context
class ConversationMemory:
    def __init__(self, max_history=5):
        self.history = deque(maxlen=max_history)
        
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

# Initialize conversation memory
memory = ConversationMemory(max_history=5)

# Function to embed text using Gemini's `text-embedding-004` model
def embed_text_with_gemini(text: str) -> List[float]:
    if text.strip():
        response = genai.embed_content(
            model='text-embedding-004',
            content=text,
            task_type="retrieval_document"
        )
        return response['embedding']
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

def chat_with_harry_potter_bot(user_message: str) -> str:
    """Process a user message and generate a conversational response"""
    # Get conversation history
    conversation_context = memory.get_conversation_context()
    
    # Special commands
    if user_message.lower() == "clear history":
        memory.clear()
        return "Obliviate! Our conversation history has been cleared."
    
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
    
    return response

    # Interactive chat loop with markdown rendering
def interactive_chat():
    try:
        # Try to import rich for markdown rendering
        from rich.console import Console
        from rich.markdown import Markdown
        has_rich = True
        console = Console()
    except ImportError:
        has_rich = False
        print("\nTip: Install 'rich' package for better markdown rendering: pip install rich\n")

    print("\n🧙‍♂️ Welcome to the Harry Potter Chat Bot! 🧙‍♀️")
    print("Ask me anything about Harry Potter and the Sorcerer's Stone!")
    print("(Type 'exit' to quit or 'clear history' to reset our conversation)\n")
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() == "exit":
            print("\nThanks for chatting! Mischief managed! ✨")
            break
            
        response = chat_with_harry_potter_bot(user_input)
        
        # Display response with markdown rendering if available
        print()
        if has_rich:
            console.print(Markdown(response))
        else:
            print(f"Harry Potter Bot: {response}")
        print()

# Example usage with sample questions
if __name__ == "__main__":
    interactive_chat()