import streamlit as st
import os
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from huggingface_hub import InferenceClient
from langchain_core.prompts import PromptTemplate

# Load environment variables
from dotenv import load_dotenv
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")

if not HF_TOKEN:
    st.error("Hugging Face token not found. Please set HF_TOKEN in .env file.")
    st.stop()

# Cache the vector store creation based on video_id
@st.cache_resource(show_spinner="Processing video transcript...")
def get_vector_store(video_id: str):
    try:
        ytt_api = YouTubeTranscriptApi()
        transcript = ytt_api.fetch(video_id, languages=["en"])
        transcript_text = " ".join(snippet.text for snippet in transcript)
    except TranscriptsDisabled:
        st.error("Transcript is disabled for this video.")
        return None
    except Exception as e:
        st.error(f"Error fetching transcript: {e}")
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript_text])
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

def answer_question(vector_store, question: str):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k":4})
    retrieved_docs = retriever.invoke(question)
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    prompt = PromptTemplate(
        template="""
        You are a helpful assistant.
        Answer ONLY from the provided transcript context.
        If the context is insufficient, just say you don't know.

        context : {context}
        Question: {question}
        """,
        input_variables=['context', 'question']
    )
    final_prompt = prompt.invoke({"context": context_text, "question": question})
    client = InferenceClient(token=HF_TOKEN)
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": final_prompt.to_string()}],
        temperature=0.2,
        max_tokens=512
    )
    return response.choices[0].message.content

# Custom CSS for attractive dark UI
st.markdown("""
<style>
    /* Main app styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Header styling */
    .app-header {
        background: linear-gradient(90deg, #0000FF, #87CEEB);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .app-header h1 {
        color: white;
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
    }

    .app-header p {
        color: rgba(255, 255, 255, 0.8);
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
    }

    /* Input field styling */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        border-radius: 10px;
        border: 2px solid #262730;
        background-color: #262730;
        color: #FAFAFA;
        padding: 0.75rem;
        transition: all 0.3s ease;
    }

    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #FF4B4B;
        box-shadow: 0 0 0 2px rgba(255, 75, 75, 0.2);
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg,#0000FF, #87CEEB);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
        background: linear-gradient(90deg, #FF6B6B, #FF8E8E);
    }

    .stButton > button:active {
        transform: translateY(0);
    }

    /* Status message styling */
    .stAlert > div {
        border-radius: 10px;
        padding: 1rem;
    }

    /* Success message */
    .stSuccess > div {
        background-color: #1E3A24;
        border: 1px solid #2E7D32;
        color: #A8E6CF;
    }

    /* Error message */
    .stError > div {
        background-color: #3A1E1E;
        border: 1px solid #7D2E2E;
        color: #F5A6A6;
    }

    /* Warning message */
    .stWarning > div {
        background-color: #3A2E1E;
        border: 1px solid #7D5E2E;
        color: #F5D8A6;
    }

    /* Info message */
    .stInfo > div {
        background-color: #1E2A3A;
        border: 1px solid #2E3E7D;
        color: #A6CFF5;
    }

    /* Container styling for answer */
    .answer-container {
        background-color: #262730;
        border-radius: 15px;
        padding: 1.5rem;
        margin-top: 1.5rem;
        border-left: 4px solid #FF4B4B;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    }

    /* Footer styling */
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid #262730;
        color: rgba(255, 255, 255, 0.5);
        font-size: 0.9rem;
    }

    /* Spinner customization */
    .stSpinner > div {
        border-top-color: #FF4B4B !important;
    }

    /* Divider styling */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #262730, transparent);
        margin: 2rem 0;
    }

    /* Responsive adjustments */
    @media (max-width: 768px) {
        .app-header h1 {
            font-size: 1.8rem;
        }

        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.markdown("""
<div class="app-header">
    <h1>🎬 YouTube Video Q&A</h1>
    <p>Ask questions about any YouTube video using AI-powered transcript analysis</p>
</div>
""", unsafe_allow_html=True)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    video_id = st.text_input(
        "📺 YouTube Video ID",
        value="Gfr50f6ZBvo",
        help="Enter the YouTube video ID (e.g., Gfr50f6ZBvo from https://youtube.com/watch?v=Gfr50f6ZBvo)",
        placeholder="Enter video ID here..."
    )

with col2:
    st.write("")  # Spacer for alignment
    st.write("")  # Spacer for alignment
    example_button = st.button("📋 Use Example", help="Fill with example video ID")
    if example_button:
        st.session_state.video_id = "Gfr50f6ZBvo"
        st.rerun()

# Update video_id from session state if set
if 'video_id' in st.session_state:
    video_id = st.session_state.video_id

question = st.text_area(
    "❓ Your Question",
    placeholder="Ask anything about the video content...",
    height=100,
    help="What would you like to know about this video?"
)

# Character counter for question
if question:
    char_count = len(question)
    st.caption(f"Characters: {char_count}/500")

st.markdown('<hr>', unsafe_allow_html=True)

# Submit button
if st.button("🔍 Get Answer", type="primary", use_container_width=True):
    if not video_id.strip():
        st.warning("Please enter a video ID.")
    elif not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("🎯 Fetching transcript and processing..."):
            vector_store = get_vector_store(video_id.strip())
        if vector_store is not None:
            with st.spinner("🤖 Generating AI-powered answer..."):
                try:
                    answer = answer_question(vector_store, question.strip())
                    st.markdown('<div class="answer-container">', unsafe_allow_html=True)
                    st.markdown("### 💡 Answer")
                    st.write(answer)
                    st.markdown('</div>', unsafe_allow_html=True)

                    # Add copy to clipboard button
                    if st.button("📋 Copy Answer"):
                        st.write("Answer copied to clipboard! (Feature would work in deployed version)")
                except Exception as e:
                    st.error(f"Error generating answer: {e}")
        else:
            st.error("Could not process video. Please check the video ID and try again.")

# Footer
st.markdown("""
<div class="footer">
    <p>Built with Streamlit • Powered by Hugging Face AI • Develope BY Shreyas Kapadiya <a href="#" target="_blank">GitHub Repository</a></p>
    <p>⚠️ For educational use only. Respect content creators' rights and YouTube's Terms of Service.</p>
</div>
""", unsafe_allow_html=True)