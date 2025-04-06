import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"

from langchain_community.embeddings import HuggingFaceEmbeddings
import streamlit as st
from gtts import gTTS
import pygame
import io
import speech_recognition as sr
from audio_recorder_streamlit import audio_recorder
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import matplotlib.pyplot as plt

PDF_PATHS = {
    "SQL": "Copy of SQL-EN.pdf",
    "Python": "PYTHON PROGRAMMING NOTES.pdf",
    "Statistics": "1208327573299-12059958653-ticket 2.pdf"
}

URLS = {
    "powerbi": "https://learn.microsoft.com/en-us/power-bi/",
    "machine_learning": "https://www.geeksforgeeks.org/machine-learning-algorithms-cheat-sheet/",
    "deep_learning": "https://www.geeksforgeeks.org/deep-learning-tutorial/",
    "generative_ai": "https://www.gartner.com/en/topics/generative-ai"
}

DB_DIR = "faiss_db"

def speak(text):
    tts = gTTS(text=text, lang='en')
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    pygame.mixer.init()
    pygame.mixer.music.load(fp)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

# Voice input using audio_recorder + SpeechRecognition
def record_and_transcribe():
    st.write("🎙️ Click the mic to record (10 seconds)...")
    audio_bytes = audio_recorder(pause_threshold=2.0)
    if audio_bytes:
        recognizer = sr.Recognizer()
        with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
            audio = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            return "Sorry, I could not understand the audio."
    return "No audio recorded."

def get_embeddings(api_key):
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_key
    return HuggingFaceEmbeddings()

def get_llm(model, api_key):
    os.environ["Groq_api_key"] = api_key
    return ChatGroq(temperature=0.5, model_name=model)

# [keywords_list dictionary remains unchanged – omitted for brevity]
keywords_list = {
    "sql":  [
    "SELECT", "JOIN", "INNER JOIN", "LEFT JOIN", "GROUP BY", "ORDER BY", "WHERE", "HAVING", 
    "DISTINCT", "COUNT", "SUM", "AVG", "MAX", "MIN", "SUBQUERY", "CTE", "UNION", "PRIMARY KEY", 
    "FOREIGN KEY", "INDEX", "VIEW", "TRUNCATE", "DELETE", "INSERT", "UPDATE", "ALTER", "CREATE", 
    "DROP", "NULL", "IS NULL", "NOT NULL"
]
,
    "python":  [
    "list", "tuple", "dictionary", "loop", "for loop", "while loop", "function", "lambda", 
    "class", "object", "inheritance", "try-except", "file handling", "comprehension", "decorator",
    "generator", "iterator", "module", "package", "pandas", "numpy", "matplotlib", "scikit-learn",
    "dataframe", "regex", "json", "virtual environment", "pip", "OOP", "mutable", "immutable"
]
,
    "statistics":  [
    "mean", "median", "mode", "variance", "standard deviation", "probability", "distribution", 
    "normal distribution", "z-score", "t-test", "chi-square", "hypothesis testing", "null hypothesis", 
    "alternative hypothesis", "confidence interval", "p-value", "correlation", "causation", 
    "outlier", "skewness", "kurtosis", "sample", "population", "central limit theorem", 
    "bayesian", "standard error", "regression", "anova"
]
,
    "powerbi":  [
    "dashboard", "report", "visual", "DAX", "Power Query", "data model", "measure", "calculated column", 
    "filter", "slicer", "table", "matrix", "bar chart", "line chart", "card", "map", "relationship", 
    "data source", "refresh", "Power BI Desktop", "Power BI Service", "drill down", "bookmark", 
    "tooltip", "KPIs", "RLS", "aggregation", "dataset"
]
,
    "machine_learning" : [
    # Core Concepts
    "model", "training", "testing", "validation", "accuracy", "regression", "classification", 
    "supervised learning", "unsupervised learning", "semi-supervised learning", "reinforcement learning",
    "clustering", "overfitting", "underfitting", "bias", "variance", "cross-validation",
    "feature selection", "feature engineering", "dimensionality reduction", "normalization", "scaling",
    
    # Metrics
    "confusion matrix", "precision", "recall", "F1 score", "ROC curve", "AUC", 
    "mean squared error", "mean absolute error", "R2 score", "log loss", "silhouette score",
    
    # Algorithms (Regression)
    "linear regression", "ridge regression", "lasso regression", "elastic net",
    
    # Algorithms (Classification)
    "logistic regression", "decision tree", "random forest", "support vector machine", "SVM",
    "naive bayes", "K-nearest neighbors", "KNN", "gradient boosting", "XGBoost", "LightGBM", 
    "CatBoost", "neural network",
    
    # Algorithms (Clustering)
    "k-means", "hierarchical clustering", "DBSCAN", "GMM", "agglomerative clustering",
    
    # Ensemble Methods
    "ensemble", "bagging", "boosting", "stacking", "voting classifier",
    
    # Optimization
    "gradient descent", "stochastic gradient descent", "batch gradient descent", 
    "learning rate", "loss function", "cost function", "objective function",
    
    # Hyperparameters
    "hyperparameter tuning", "grid search", "random search", "Bayesian optimization",
    
    # Target Variable Types
    "continuous", "categorical", "binary", "multiclass", "ordinal", "nominal",
    
    # Data Types and Preprocessing
    "numerical", "categorical", "missing values", "outliers", "label encoding", 
    "one-hot encoding", "scaling", "standardization", "normalization",
    
    # Data Handling
    "train test split", "stratified sampling", "resampling", "under sampling", "over sampling",
    "SMOTE", "cross-validation", "k-fold validation", "time series split",

    # Advanced Topics
    "model interpretability", "feature importance", "SHAP", "LIME", "autoML", 
    "pipeline", "MLops", "model deployment", "drift", "retraining"
]
,
    "deep_learning" : [
    "neural network", "activation function", "backpropagation", "gradient descent", "ReLU", "sigmoid", 
    "softmax", "dropout", "batch normalization", "CNN", "RNN", "LSTM", "GAN", "transformer", 
    "attention", "loss function", "optimizer", "learning rate", "epoch", "batch size", "training set", 
    "test set", "overfitting", "underfitting", "autoencoder", "embedding", "convolution", 
    "pooling", "sequence model"
],

    "generative_ai":  [
    "LLM", "prompt", "generation", "token", "transformer", "autoregressive", "GPT", "sampling", 
    "temperature", "top-k", "top-p", "fine-tuning", "LoRA", "text generation", "image generation", 
    "RAG", "embedding", "context window", "inference", "hallucination", "zero-shot", 
    "few-shot", "chain of thought", "agents", "vectorstore", "prompt engineering", 
    "semantic search", "embedding model"
]

}

# ... (Insert the full `keywords_list` dictionary here as it was previously given)

def evaluate_response(question, response):
    vec = TfidfVectorizer().fit_transform([question, response])
    score = cosine_similarity(vec[0:1], vec[1:2])[0][0]
    found_keywords = []
    for key, keywords in keywords_list.items():
        for kw in keywords:
            if kw.lower() in response.lower():
                found_keywords.append((key, kw))
    return {
        "similarity_score": round(score, 2),
        "keywords": found_keywords,
        "summary": f"Similarity: {round(score, 2)}, Keywords Matched: {len(found_keywords)}"
    }

def generate_report(feedbacks):
    total_score = sum(f['similarity_score'] for f in feedbacks) / len(feedbacks)
    all_keywords = [kw for f in feedbacks for kw in f['keywords']]
    topic_count = {}
    for topic, _ in all_keywords:
        topic_count[topic] = topic_count.get(topic, 0) + 1
    weakest_topic = min(topic_count.items(), key=lambda x: x[1])[0] if topic_count else "various topics"
    keywords_used = [kw for _, kw in all_keywords]

    technical_strength = "Good" if len(keywords_used) >= 10 else "Basic"
    communication_score = total_score

    improvement_area = []
    if communication_score < 0.5:
        improvement_area.append("communication skills")
    if technical_strength == "Basic":
        improvement_area.append("technical knowledge")

    fig, ax = plt.subplots()
    ax.bar(["Communication", "Technical"], [communication_score * 100, len(keywords_used)])
    ax.set_ylim(0, max(100, len(keywords_used) + 10))
    ax.set_ylabel("Score / Count")
    ax.set_title("Candidate Performance Breakdown")

    st.subheader("📊 Final Interview Report")
    st.pyplot(fig)

    summary_text = f"""
    **Overall Similarity Score:** {round(total_score, 2)}  
    **Matched Keywords:** {keywords_used}  
    **Technical Knowledge:** {technical_strength}  
    **Communication Skills Score:** {round(communication_score * 100, 1)}%  
    **Areas to Improve:** {', '.join(improvement_area) if improvement_area else 'None – Great job!'}
    """

    spoken_feedback = f"""
    {st.session_state.name}, your technical knowledge is {technical_strength.lower()} and your communication score is {round(communication_score * 100)} percent.
    You need to improve in {', and '.join(improvement_area)}.
    """

    speak(spoken_feedback)

    return {
        "text": summary_text,
        "weak_area": ', '.join(improvement_area) if improvement_area else "None"
    }

def build_and_save_vectorstore():
    pdf_docs, web_docs = [], []

    for topic, path in PDF_PATHS.items():
        loader = PyPDFLoader(path)
        docs = loader.load()
        for doc in docs:
            doc.metadata["topic"] = topic.lower()
        pdf_docs.extend(docs)

    for topic, url in URLS.items():
        loader = WebBaseLoader(url)
        docs = loader.load()
        for doc in docs:
            doc.metadata["topic"] = topic.lower()
        web_docs.extend(docs)

    all_docs = pdf_docs + web_docs
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(all_docs)

    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(DB_DIR)
    return vectorstore

@st.cache_resource
def load_vectorstore():
    if os.path.exists(DB_DIR):
        return FAISS.load_local(DB_DIR, HuggingFaceEmbeddings(), allow_dangerous_deserialization=True)
    else:
        return build_and_save_vectorstore()

# Streamlit UI
st.set_page_config(page_title="AI Interview Chatbot", layout="wide")
st.title("🎙️ AI Interviewer Chatbot ")

if "name" not in st.session_state:
    st.session_state.name = ""
    st.session_state.course = None
    st.session_state.topics = []
    st.session_state.questions = []
    st.session_state.answers = []
    st.session_state.feedback = []

with st.sidebar:
    st.header("🔧 Configuration")
    api_key = st.text_input("Enter Groq/HuggingFace API Key", type="password")
    model_choice = st.selectbox("Groq Model", ["mixtral-8x7b-32768", "llama2-70b-4096", "gemma2-7b-it", "gemma2-9b-it"])

if not st.session_state.name:
    st.subheader("👋 Introduce yourself?")
    if st.button("🎤 Speak Your Name"):
        user_name = record_and_transcribe()
        st.session_state.name = user_name
        st.success(f"Nice to meet you, {user_name}!")
        speak(f"Hello {user_name}, nice to meet you please select the course from below deop downlist!")

if st.session_state.name:
    course = st.selectbox("Select a course:", ["Data Analyst", "Data Scientist"])
    if course:
        st.session_state.course = course
        topics = ["sql", "python", "powerbi", "statistics"] if course == "Data Analyst" else ["sql", "python", "powerbi", "machine_learning", "deep_learning", "generative_ai"]
        selected_topics = st.multiselect("Choose topics to be interviewed on:", topics)
        st.session_state.topics = selected_topics

if st.button("🎤 Start Interview") and st.session_state.topics:
    embeddings = get_embeddings(api_key)
    llm = get_llm(model_choice, api_key)

    try:
        vectordb = load_vectorstore()
        for topic in st.session_state.topics:
            retriever = vectordb.as_retriever(search_kwargs={"filter": {"topic": topic}})
            qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

            speak(f"{st.session_state.name}, let's begin your interview on {topic}.")
            st.info(f"Interview started on **{topic}**. Answer using your voice.")

            for i in range(1, 6):
                prompt = (f"Generate a level {i} interview question from the topic: {topic}. "
                          f"Ensure that the question is unique and increases in difficulty. "
                          f"Do not repeat previously asked questions.")
                question = qa.run(prompt)
                st.session_state.questions.append(question)

                speak(question)
                st.markdown(f"**Q{i} ({topic}):** {question}")
                st.write("🎧 Listening for your answer...")
                response_text = record_and_transcribe()
                st.write(f"🗣️ Your Answer: {response_text}")

                feedback = evaluate_response(question, response_text)
                st.session_state.answers.append(response_text)
                st.session_state.feedback.append(feedback)
                st.success(f"✅ Feedback: {feedback['summary']}")

        report = generate_report(st.session_state.feedback)
        st.markdown(report["text"])

    except Exception as e:
        st.error(f"Error during interview: {e}")
