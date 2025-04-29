import os
import google.generativeai as genai
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import re
from typing import List, Dict, Any

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

# Improved prompt engineering for better answers
def generate_answer_from_context(context: str, question: str) -> str:
    try:
        # Initialize the model
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Optimized prompt combining book context with AI knowledge
        prompt = f"""As an expert on Harry Potter, answer this question about 'Harry Potter and the Sorcerer's Stone': "{question}"

        Book context to incorporate:
        {context}

        Please create a single, cohesive answer that:
        1. Seamlessly integrates information from both the provided book context and your knowledge of Harry Potter
        2. Remains concise while being complete and accurate
        3. Addresses the question directly with the most relevant details
        4. Maintains the authentic voice and style of the Harry Potter world
        5. Prioritizes book-accurate information but fills gaps naturally when needed 

        Your accurate answer:"""
        
        # Generate the response with balanced settings for accuracy and brevity
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.2,  # Slightly higher temperature for more complete answers
                top_p=0.9,
                top_k=45,
                max_output_tokens=1500  # Allow slightly more tokens for accuracy
            )
        )
        
        return response.text.strip()
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        return "Sorry, I encountered an error while generating the response."

def answer_harry_potter_question(question: str) -> str:
    """Main function to answer questions about Harry Potter"""
    print(f"Question: {question}")
    
    # 1. Retrieve relevant passages
    passages = retrieve_relevant_documents(question, top_k=30)  # Increased to get more context
    
    if not passages:
        return "I couldn't find any relevant information to answer your question about Harry Potter."
    
    # 2. Build optimized context
    context = build_context(passages)
    
    # 3. Generate answer
    answer = generate_answer_from_context(context, question)
    
    return answer

# Example usage
if __name__ == "__main__":
    # This can be any question about Harry Potter and the Sorcerer's Stone
    question = "tell me the rules and regulation for playing qudich"
    
    answer = answer_harry_potter_question(question)
    print(f"\nAI Answer: {answer}")