import os
import time
import google.generativeai as genai
import fitz  # PyMuPDF for PDF extraction
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# Load environment variables from .env file
load_dotenv()

# Initialize Pinecone client using the new way
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Define Serverless specifications
serverless_spec = ServerlessSpec(
    cloud="aws",       # You can change this to other providers if needed
    region="us-east-1"  # You can change this region as well
)

# Index name
index_name = "harry-potter-and-the-sorcerers-stone"  # Name of the index to be created

# Create index if it doesn't exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768,  # Dimension based on Gemini embedding (ensure to confirm if it's 768)
        metric="cosine",  # Common metric for embeddings
        spec=serverless_spec  # Use the serverless specification for the index
    )

# Connect to the index
index = pc.Index(index_name)

# Initialize Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Function to extract text from a PDF file
def extract_pdf_text(pdf_path):
    doc = fitz.open(pdf_path)  # Open the PDF
    pdf_text = []
    
    # Extract text page by page
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)  # Load each page
        text = page.get_text()  # Extract text from page
        
        if text.strip():  # Check if the text is not empty or just whitespace
            pdf_text.append(text)
        else:
            pdf_text.append('')  # Append empty string for pages with no text
    
    return pdf_text, doc.metadata.get("title", "Unknown Title")  # Return text and title (if available)

# Function to embed text using Gemini's `text-embedding-004` model
def embed_text_with_gemini(text):
    if text.strip():  # Ensure the text is not empty or whitespace
        response = genai.embed_content(
            model='text-embedding-004',
            content=text,
            task_type="retrieval_document"  # Specify the task type for better embeddings
        )
        return response['embedding']
    else:
        print("Warning: Skipping empty page.")
        return None  # Return None if the page text is empty

# Function to upload PDF pages to Pinecone with metadata
def upload_pdf_to_pinecone(pdf_path):
    pdf_text, pdf_title = extract_pdf_text(pdf_path)
    
    # Loop through each page and upsert its embedding to Pinecone
    for page_num, text in enumerate(pdf_text):
        embedding_vector = embed_text_with_gemini(text)
        
        if embedding_vector is not None:  # Only upsert if the embedding is valid
            page_id = f"page-{page_num + 1}"  # Use page number as unique ID
            
            # Metadata to add with the embedding
            metadata = {
                "pdf_title": pdf_title,  # Title of the PDF document
                "page_number": page_num + 1 , # Page number
                "text": text[:8000],  # Truncate for metadata
                "char_count": len(text)
            }
            
            # Upsert the page embedding into Pinecone
            index.upsert([
                (page_id, embedding_vector, metadata)  # Including metadata
            ])
            print(f"Page {page_num + 1} successfully embedded and upserted with metadata.")
        else:
            print(f"Skipping Page {page_num + 1} as it contains no valid text.")
# Example usage: Uploading a PDF to Pinecone
pdf_path = "D:\\Harry Poter Bot\\harry potter books\\HP1 - Harry Potter and the Sorcerer_s Stone.pdf"  # Path to your PDF file
upload_pdf_to_pinecone(pdf_path)

# Success message
print("All pages from the PDF have been successfully embedded and upserted into Pinecone.")