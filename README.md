# 🎙️ AI Interview Chatbot using Streamlit, Groq LLMs & HuggingFace Embeddings

## 📌 Project Description

This AI Interview Chatbot is an interactive, voice-enabled interviewer built using **Streamlit**. It conducts interviews based on selected topics from courses such as **Data Analyst** and **Data Scientist**, simulating a real-time assessment environment. The chatbot dynamically generates questions using **Groq-hosted LLMs**, evaluates spoken responses based on **semantic similarity** and **keyword matching**, and provides detailed **performance reports** with voice feedback and visualizations.

---

## 🎯 Aim of the Project

- Provide an AI-based mock interview platform.
- Evaluate technical and communication skills.
- Generate questions dynamically based on selected topics.
- Accept responses through voice or text.
- Score answers using similarity metrics and keyword analysis.
- Deliver voice feedback and performance visualization.

---

## 🧠 Core Functionalities

- 🎤 **Voice Input & Output** (via `speech_recognition`, `pyttsx3`, `gTTS`, `pygame`)
- 📚 **RAG Pipeline** using:
  - PDFs for topics like SQL, Python, Statistics
  - Web URLs for topics like Power BI, Machine Learning, Deep Learning, Generative AI
- 🧠 **Dynamic Question Generation** using Groq LLMs
- ✅ **Answer Evaluation** with:
  - TF-IDF + Cosine Similarity
  - Domain Keyword Matching
- 📊 **Performance Report** with:
  - Score breakdown
  - Visual bar charts (via `matplotlib`)
  - Voice-based final feedback

---

## 🔧 Libraries & Tools Used

| Category             | Libraries Used |
|----------------------|----------------|
| **UI & App Framework** | `streamlit`, `matplotlib` |
| **Voice Interaction** | `speech_recognition`, `pyttsx3`, `gTTS`, `pygame` |
| **Embeddings**       | `HuggingFaceEmbeddings` from `langchain_community` |
| **LLMs**             | `ChatGroq` from `langchain_groq` |
| **RAG Pipeline**     | `FAISS`, `PyPDFLoader`, `WebBaseLoader`, `RecursiveCharacterTextSplitter` |
| **Evaluation**       | `TfidfVectorizer`, `cosine_similarity` from `sklearn` |

---

## 📂 Vector Database

- Vector store built using:
  - `langchain.vectorstores.FAISS`
  - Document sources: PDFs + URLs
- Saved locally to `faiss_db/` directory
- Reused via cached loading in Streamlit

---

## 📊 Evaluation Logic

1. **Similarity Score** via `TfidfVectorizer` + `cosine_similarity`
2. **Keyword Match** based on domain-specific keyword dictionaries
3. **Report Includes**:
   - Communication skill score
   - Technical keyword match count
   - Weakest topic
   - Strengths and areas of improvement
   - Visual bar graph of performance

---

## 🧠 LLMs & Embeddings Used

- **LLM Provider**: [Groq Cloud](https://console.groq.com/)
  - Models Supported:
    - `mixtral-8x7b-32768`
    - `llama2-70b-4096`
    - `gemma2-7b-it`
    - `gemma2-9b-it`

- **Embeddings**: `HuggingFaceEmbeddings` from `langchain_community`

---

## 📘 How it Works (Flow)

1. User speaks their name (voice input).
2. Selects course and preferred topics.
3. Interview begins — 5 questions per topic, difficulty increases progressively.
4. Answers are recorded via microphone (10 sec limit).
5. System evaluates responses.
6. Final report and spoken feedback generated.


---

## 🚀 To Run the Project

```bash
pip install -r requirements.txt
streamlit run main.py


