# 📄 RAG PDF Q&A Chatbot with Conversation Memory

A Retrieval-Augmented Generation (RAG) based chatbot that allows users to ask questions from PDF documents while maintaining chat history for contextual responses.

---

## 📌 Overview

This project enables intelligent question-answering over PDF files using a combination of:
- Vector database retrieval
- Large Language Models (LLMs)
- Conversation memory

The chatbot retrieves relevant chunks from the PDF and generates accurate, context-aware answers.

---

## 🚀 Features

- 📄 Upload and process PDF documents  
- 🔍 Semantic search using vector embeddings  
- 💬 Chat-based Q&A interface  
- 🧠 Maintains conversation history (context-aware responses)  
- ⚡ Fast retrieval using ChromaDB  
- 🔗 Integrated with LangChain for RAG pipeline  

---

## 🏗️ Architecture

```
PDF → Text Chunking → Embeddings → Vector DB (Chroma)
                                     ↓
User Query → Embedding → Similarity Search → LLM → Response
                                     ↑
                              Chat History Memory
```

---

## ⚙️ Tech Stack

- **Language:** Python  
- **Framework:** LangChain  
- **Vector DB:** ChromaDB  
- **LLM:** OpenAI / compatible LLM  
- **Embeddings:** OpenAI Embeddings  
- **Interface:** CLI / Streamlit (optional)  

---

## 📂 Project Structure

```
.
├── app.py              # Main application
├── chroma_db/          # Vector database storage
├── requirements.txt    # Dependencies
├── temp.pdf            # Sample PDF
└── README.md
```

---

## 🧠 How It Works

1. PDF is loaded and split into smaller chunks  
2. Each chunk is converted into vector embeddings  
3. Embeddings are stored in ChromaDB  
4. User query is converted into embedding  
5. Similar chunks are retrieved using semantic search  
6. LLM generates answer using:
   - Retrieved context  
   - Chat history  

---

## ▶️ How to Run Locally

### 1. Clone the repository
```
git clone https://github.com/abhisinghh72/RAG-QnA-Conversation-With-PDF-Including-Chat-History.git
cd RAG-QnA-Conversation-With-PDF-Including-Chat-History
```

### 2. Create virtual environment
```
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```
pip install -r requirements.txt
```

### 4. Add environment variables
Create `.env` file:
```
OPENAI_API_KEY=your_api_key
```

### 5. Run the app
```
python app.py
```


---

## 🎯 Use Cases

- Document-based Q&A systems  
- Research paper assistant  
- Legal/financial document analysis  
- Knowledge base chatbot  

---

## 📈 Future Improvements

- Web UI using Streamlit or React  
- Support multiple PDFs  
- Deploy on cloud (AWS / Render / HuggingFace Spaces)  
- Add authentication & user sessions  

---

## 👤 Author

Abhishek Singh
