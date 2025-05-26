## 🧙‍♂️ Harry-Potter-AI-Bot

An AI-powered conversational bot infused with the magical world of **Harry Potter**, built by embedding book knowledge into **Pinecone** and giving the AI a unique fan-like personality. Ask it anything about the wizarding world — from deep lore to character facts — and get accurate, personality-rich responses like you're chatting with a true Hogwarts expert.

---

### 🧠 Project Overview

**Harry-Potter-AI-Bot** is a fan-focused conversational AI trained using the content of the official Harry Potter books. Using **OpenAI** for natural language understanding and **Pinecone vector database** for semantic retrieval, the bot answers any question related to the Harry Potter universe.

When book content is insufficient, the AI fills gaps using its general language model capabilities — always staying in character as a passionate Harry Potter enthusiast.

---

### ✨ Features

- 🧠 Pinecone-powered search of embedded book content
- 💬 Conversational memory with GPT-based responses
- 🎭 AI with a custom Harry Potter fan personality
- 📚 Accurate answers grounded in official book lore
- 🔍 Fallback to LLM general knowledge when needed
- 🧵 Persistent chat experience (optional)

---

### 🔮 Magic Under the Hood

- **Embedding**: Book content is split into chunks and embedded using `text-embedding-ada-002`
- **Storage**: Embeddings are stored in Pinecone for semantic vector search
- **LLM**: OpenAI GPT-4 or GPT-3.5 handles chat completion
- **Prompt Engineering**: Custom system prompt gives AI a witty, knowledgeable Harry Potter fan persona

---

### 🧰 Tech Stack

- **Language**: Python  
- **AI Model**: OpenAI GPT-4  
- **Vector DB**: Pinecone  
- **Embeddings**: `text-embedding-ada-002`  
- **Frontend**: Streamlit or Flask (optional)  
- **Prompt**: Persona-enforced, Hogwarts-themed context  

---

### 🚀 Getting Started

#### Prerequisites

- Python 3.8+
- OpenAI API Key
- Pinecone API Key
- Harry Potter book text (cleaned, licensed for personal use)

#### Setup


    git clone https://github.com/your-username/Harry-Potter-AI-Bot.git
    cd Harry-Potter-AI-Bot
    pip install -r requirements.txt


#### Configure .env

    OPENAI_API_KEY=your-openai-key
    PINECONE_API_KEY=your-pinecone-key
    PINECONE_ENV=us-west1-gcp
    INDEX_NAME=harry-potter-index


### 🔮 Future Enhancements

- 🗃️ Memory for multi-turn conversation
- 🧭 Timeline-based events and chapter reference
- 🧙 Voice chat integration with character voices
- 🎓 Hogwarts House selector with quizzes

### 📄 License

This project is for educational and fan purposes only. Not affiliated with J.K. Rowling, Warner Bros., or the official Harry Potter franchise. Licensed under MIT License.
