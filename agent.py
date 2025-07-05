import os
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from llama_index.llms.groq import Groq
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Document
from llama_index.core.tools import QueryEngineTool, FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.memory.mem0 import Mem0Memory
import requests
import tempfile
import shutil
from pathlib import Path
import datetime

# Page configuration
st.set_page_config(
    page_title="Personal AI Assistant",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with enhanced search bar styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Main background - softer gradient with better contrast */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #f5576c 75%, #4facfe 100%);
        background-attachment: fixed;
        min-height: 100vh;
    }
    
    /* Enhanced Search Bar Container */
    .search-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 50px;
        border: 4px solid transparent;
        background-clip: padding-box;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.2);
        padding: 2rem;
        margin: 2rem 0;
        position: relative;
        overflow: hidden;
    }
    
    .search-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, #667eea, #764ba2, #f093fb, #f5576c, #4facfe);
        border-radius: 50px;
        z-index: -1;
        padding: 4px;
        mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
        mask-composite: subtract;
    }
    
    /* Search Input Styling */
    .search-input-wrapper {
        position: relative;
        display: flex;
        align-items: center;
        background: rgba(255, 255, 255, 0.9);
        border-radius: 35px;
        padding: 0.5rem;
        border: 3px solid transparent;
        background-clip: padding-box;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .search-input-wrapper:hover {
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.3);
        transform: translateY(-2px);
    }
    
    .search-input-wrapper::before {
        content: '';
        position: absolute;
        top: -3px;
        left: -3px;
        right: -3px;
        bottom: -3px;
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 35px;
        z-index: -1;
    }
    
    .search-icon {
        font-size: 1.5rem;
        color: #667eea;
        margin: 0 1rem;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.1); }
        100% { transform: scale(1); }
    }
    
    /* Chat Input Enhancements */
    .stChatInput {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        margin: 0 !important;
        box-shadow: none !important;
    }
    
    .stChatInput > div {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    
    .stChatInput input {
        background: transparent !important;
        border: none !important;
        color: #2c3e50 !important;
        font-weight: 600 !important;
        font-size: 1.2rem !important;
        padding: 1rem 1.5rem !important;
        outline: none !important;
        flex: 1;
        border-radius: 0 !important;
    }
    
    .stChatInput input::placeholder {
        color: #74b9ff !important;
        font-weight: 500 !important;
    }
    
    /* Search Button */
    .search-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        border-radius: 50%;
        width: 50px;
        height: 50px;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
        margin-right: 0.5rem;
    }
    
    .search-button:hover {
        transform: scale(1.1);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5);
    }
    
    .search-button svg {
        width: 20px;
        height: 20px;
        fill: white;
    }
    
    /* Search Suggestions */
    .search-suggestions {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 1rem;
        justify-content: center;
    }
    
    .suggestion-chip {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
        border: 2px solid rgba(102, 126, 234, 0.3);
        border-radius: 20px;
        padding: 0.5rem 1rem;
        font-size: 0.9rem;
        font-weight: 500;
        color: #2c3e50;
        cursor: pointer;
        transition: all 0.3s ease;
        white-space: nowrap;
    }
    
    .suggestion-chip:hover {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
    }
    
    /* Enhanced search label */
    .search-label {
        text-align: center;
        font-size: 1.1rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 1rem;
        opacity: 0.8;
    }
    
    /* Sidebar styling - dark with high contrast */
    .css-1d391kg {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 0 25px 25px 0;
        border-right: 3px solid #ff6b6b;
        box-shadow: 5px 0 20px rgba(0, 0, 0, 0.3);
    }
    
    /* Main content area - high contrast white background */
    .main .block-container {
        background: rgba(255, 255, 255, 0.98);
        backdrop-filter: blur(20px);
        border-radius: 30px;
        border: 3px solid rgba(255, 255, 255, 0.4);
        padding: 3rem;
        margin-top: 2rem;
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.2);
    }
    
    /* Title styling - bold, readable gradient text */
    .main-title {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 4rem;
        font-weight: 900;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
        font-family: 'Inter', sans-serif;
        letter-spacing: -0.02em;
    }
    
    /* Subtitle styling - high contrast dark text */
    .subtitle {
        color: #2c3e50;
        text-align: center;
        font-size: 1.4rem;
        margin-bottom: 2rem;
        font-weight: 500;
        font-family: 'Inter', sans-serif;
        opacity: 0.8;
    }
    
    /* Chat messages - better contrast */
    .stChatMessage {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 2px solid rgba(102, 126, 234, 0.2);
        margin: 1rem 0;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
        color: #2c3e50 !important;
    }
    
    /* Chat message text */
    .stChatMessage p, .stChatMessage div {
        color: #2c3e50 !important;
        font-weight: 500;
        line-height: 1.6;
    }
    
    /* Buttons - high contrast with readable text */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        border-radius: 25px;
        color: white !important;
        font-weight: 700;
        padding: 1rem 2rem;
        transition: all 0.3s ease;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.3);
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 0.9rem;
        font-family: 'Inter', sans-serif;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Sample question buttons - better contrast */
    .stButton[data-testid*="sample"] > button {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        color: #2c3e50 !important;
        border: 2px solid #667eea;
        border-radius: 15px;
        font-size: 0.95rem;
        font-weight: 600;
        padding: 0.75rem 1.25rem;
        margin: 0.5rem 0;
        width: 100%;
        text-align: left;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.2);
        text-transform: none;
        letter-spacing: 0;
    }
    
    .stButton[data-testid*="sample"] > button:hover {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.3);
    }
    
    /* File uploader - high contrast */
    .stFileUploader {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 20px;
        border: 3px dashed #667eea;
        padding: 2rem;
        text-align: center;
        color: #2c3e50 !important;
        font-weight: 600;
    }
    
    /* Status indicators - better readability */
    .status-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 2px solid rgba(102, 126, 234, 0.3);
        padding: 1.5rem;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
    }
    
    .status-card div {
        color: #2c3e50 !important;
        font-weight: 700;
        font-size: 1.1rem;
    }
    
    /* Capability cards - improved contrast */
    .capability-card {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 2px solid rgba(102, 126, 234, 0.3);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
    }
    
    .capability-card h4 {
        color: #2c3e50 !important;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    /* Header info cards - high contrast */
    .info-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        border: 3px solid rgba(255, 255, 255, 0.4);
        padding: 1.5rem;
        text-align: center;
        color: white !important;
        font-weight: 700;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2);
    }
    
    .info-card h4 {
        color: white !important;
        font-weight: 700;
        margin: 0;
        font-size: 1.1rem;
    }
    
    /* General text styling - high contrast */
    .stMarkdown, .stText {
        color: #2c3e50 !important;
        font-weight: 500;
        line-height: 1.6;
    }
    
    /* Sidebar text - high contrast white */
    .css-1d391kg .stMarkdown {
        color: white !important;
        font-weight: 500;
    }
    
    .css-1d391kg .stMarkdown h2 {
        color: #ffd700 !important;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
        font-weight: 800;
    }
    
    .css-1d391kg .stMarkdown h4 {
        color: #ffd700 !important;
        font-weight: 700;
    }
    
    /* Success/Info messages - better contrast */
    .stSuccess {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        border: 3px solid #27ae60;
        color: #2c3e50 !important;
        font-weight: 600;
        padding: 1rem;
    }
    
    .stInfo {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        border: 3px solid #3498db;
        color: #2c3e50 !important;
        font-weight: 600;
        padding: 1rem;
    }
    
    .stError {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        border: 3px solid #e74c3c;
        color: #2c3e50 !important;
        font-weight: 600;
        padding: 1rem;
    }
    
    /* Capability grid items - high contrast */
    .capability-grid-item {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        color: white !important;
        font-weight: 700;
        margin: 0.5rem;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }
    
    .capability-grid-item:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
    }
    
    .capability-grid-item div {
        color: white !important;
        font-weight: 700;
    }
    
    /* Tools list styling - better readability */
    .tools-list {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 1.5rem;
        border: 2px solid rgba(102, 126, 234, 0.3);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    .tools-list div {
        color: #2c3e50 !important;
        font-weight: 600;
        line-height: 1.8;
    }
    
    /* Sidebar dividers */
    .css-1d391kg hr {
        border-color: rgba(255, 255, 255, 0.3);
        margin: 1.5rem 0;
    }
    
    /* Spinner styling */
    .stSpinner {
        color: #667eea;
    }
    
    /* Ensure all text is readable */
    p, div, span, h1, h2, h3, h4, h5, h6 {
        color: inherit;
        font-family: 'Inter', sans-serif;
    }
    
    /* Make sure code blocks are readable */
    .stCodeBlock, code {
        background: rgba(44, 62, 80, 0.1) !important;
        color: #2c3e50 !important;
        border-radius: 8px;
        padding: 0.5rem;
        font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
    }
    
    /* Search bar focus effects */
    .search-input-wrapper:focus-within {
        box-shadow: 0 20px 50px rgba(102, 126, 234, 0.4);
        transform: translateY(-3px);
    }
    
    .search-input-wrapper:focus-within::before {
        background: linear-gradient(135deg, #4facfe, #00f2fe);
    }
</style>
""", unsafe_allow_html=True)

# Load environment variables
@st.cache_data
def load_environment_variables():
    return {
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
        "GOOGLE_CX": os.getenv("GOOGLE_CX"),
        "GROQ_API_KEY": os.getenv("GROQ_API_KEY"),
        "MEM0_API_KEY": os.getenv("MEM0_API_KEY")
    }

env_vars = load_environment_variables()

# Model and App Settings
MODEL_NAME = "llama-3.3-70b-versatile"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_OVERLAP = 20
OUTPUT_TOKENS = 1024
CHUNK_SIZE = 1000

# Initialize LLM
@st.cache_resource
def get_llm():
    return Groq(model=MODEL_NAME, api_key=env_vars["GROQ_API_KEY"], temperature=0.7)

# Initialize Memory
@st.cache_resource
def get_memory():
    if env_vars["MEM0_API_KEY"]:
        context = {"user_id": f"streamlit_user_{st.session_state.get('user_id', 'default')}"}
        return Mem0Memory.from_client(
            api_key=env_vars["MEM0_API_KEY"], 
            context=context, 
            search_msg_limit=8
        )
    return None

# Initialize Settings
@st.cache_resource
def initialize_settings():
    Settings.llm = get_llm()
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
    Settings.num_output = OUTPUT_TOKENS
    Settings.node_parser = SentenceWindowNodeParser(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    
    # Add memory if available
    memory = get_memory()
    if memory:
        Settings.memory = memory
    
    return Settings

# Document processing functions
def process_uploaded_files(uploaded_files):
    """Process uploaded files and create documents"""
    documents = []
    temp_dir = tempfile.mkdtemp()
    
    try:
        for uploaded_file in uploaded_files:
            # Save uploaded file to temporary directory
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Read document based on file type
            if uploaded_file.type == "text/plain":
                content = uploaded_file.read().decode("utf-8")
                doc = Document(text=content, metadata={"filename": uploaded_file.name, "type": "text"})
                documents.append(doc)
            elif uploaded_file.type == "application/pdf":
                # For PDF files, use SimpleDirectoryReader
                try:
                    pdf_docs = SimpleDirectoryReader(input_files=[temp_path]).load_data()
                    for doc in pdf_docs:
                        doc.metadata["filename"] = uploaded_file.name
                    documents.extend(pdf_docs)
                except Exception as e:
                    st.error(f"Error processing PDF {uploaded_file.name}: {e}")
            else:
                # Try to read as text for other file types
                try:
                    content = uploaded_file.read().decode("utf-8")
                    doc = Document(text=content, metadata={"filename": uploaded_file.name, "type": uploaded_file.type})
                    documents.append(doc)
                except:
                    st.error(f"Unsupported file type: {uploaded_file.type}")
        
        return documents
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)

# Create index from documents
def create_document_index(documents):
    """Create vector index from documents"""
    if not documents:
        return None
    
    try:
        index = VectorStoreIndex.from_documents(documents, embed_model=Settings.embed_model)
        return index.as_query_engine()
    except Exception as e:
        st.error(f"Error creating document index: {e}")
        return None

# Utility functions (expanded for general use)
def calculate(expression: str) -> str:
    """Safely evaluate mathematical expressions"""
    try:
        # Only allow safe mathematical operations
        allowed_chars = set('0123456789+-*/.() ')
        if not all(c in allowed_chars for c in expression):
            return "Invalid characters in expression"
        
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error in calculation: {e}"

def word_count(text: str) -> str:
    """Count words, characters, and paragraphs in text"""
    words = len(text.split())
    chars = len(text)
    chars_no_spaces = len(text.replace(' ', ''))
    paragraphs = len([p for p in text.split('\n\n') if p.strip()])
    
    return f"Words: {words}, Characters: {chars}, Characters (no spaces): {chars_no_spaces}, Paragraphs: {paragraphs}"

def currency_converter(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert currency (basic rates for demo)"""
    exchange_rates = {
        ("USD", "EUR"): 0.91, ("EUR", "USD"): 1.10,
        ("USD", "GBP"): 0.79, ("GBP", "USD"): 1.27,
        ("USD", "CAD"): 1.35, ("CAD", "USD"): 0.74,
        ("USD", "JPY"): 148.0, ("JPY", "USD"): 0.0068,
        ("EUR", "GBP"): 0.87, ("GBP", "EUR"): 1.15,
        ("EUR", "CAD"): 1.48, ("CAD", "EUR"): 0.68
    }
    
    key = (from_currency.upper(), to_currency.upper())
    if key in exchange_rates:
        result = amount * exchange_rates[key]
        return f"{amount} {from_currency.upper()} = {result:.2f} {to_currency.upper()}"
    else:
        return f"Exchange rate not available for {from_currency} to {to_currency}"

def unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    """Convert between common units"""
    conversions = {
        # Length
        ("m", "ft"): 3.28084, ("ft", "m"): 0.3048,
        ("cm", "inch"): 0.393701, ("inch", "cm"): 2.54,
        ("km", "mile"): 0.621371, ("mile", "km"): 1.60934,
        
        # Weight
        ("kg", "lb"): 2.20462, ("lb", "kg"): 0.453592,
        ("g", "oz"): 0.035274, ("oz", "g"): 28.3495,
        
        # Temperature
        ("c", "f"): lambda x: (x * 9/5) + 32, ("f", "c"): lambda x: (x - 32) * 5/9,
        
        # Volume
        ("l", "gal"): 0.264172, ("gal", "l"): 3.78541,
        ("ml", "floz"): 0.033814, ("floz", "ml"): 29.5735
    }
    
    key = (from_unit.lower(), to_unit.lower())
    if key in conversions:
        converter = conversions[key]
        if callable(converter):
            result = converter(value)
        else:
            result = value * converter
        return f"{value} {from_unit} = {result:.4f} {to_unit}"
    else:
        return f"Conversion not available for {from_unit} to {to_unit}"

def get_current_time() -> str:
    """Get current date and time"""
    now = datetime.datetime.now()
    return f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}"

def google_search(query: str) -> str:
    """Search Google for information"""
    if not env_vars["GOOGLE_API_KEY"] or not env_vars["GOOGLE_CX"]:
        return "Google search not configured. Please set GOOGLE_API_KEY and GOOGLE_CX environment variables."
    
    try:
        url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={env_vars['GOOGLE_API_KEY']}&cx={env_vars['GOOGLE_CX']}"
        response = requests.get(url).json()
        if "items" in response:
            results = []
            for item in response["items"][:3]:  # Get top 3 results
                results.append(f"Title: {item['title']}\nSnippet: {item['snippet']}\nURL: {item['link']}\n")
            return "\n".join(results)
        return "No relevant information found."
    except Exception as e:
        return f"Search error: {e}"

def summarize_text(text: str, max_sentences: int = 3) -> str:
    """Simple text summarization"""
    sentences = text.split('.')
    if len(sentences) <= max_sentences:
        return text
    
    # Simple approach: take first and last sentences, and one from middle
    summary_sentences = []
    summary_sentences.append(sentences[0])
    if len(sentences) > 2:
        summary_sentences.append(sentences[len(sentences)//2])
    summary_sentences.append(sentences[-2])  # -1 is usually empty after split
    
    return '. '.join(summary_sentences) + '.'

# Initialize agent
@st.cache_resource
def initialize_agent(_document_query_engine=None):
    settings = initialize_settings()
    
    # Create general-purpose tools
    tools = [
        FunctionTool.from_defaults(fn=calculate, name="Calculator", description="Perform mathematical calculations with basic operations."),
        FunctionTool.from_defaults(fn=word_count, name="TextAnalyzer", description="Count words, characters, and paragraphs in text."),
        FunctionTool.from_defaults(fn=currency_converter, name="CurrencyConverter", description="Convert between different currencies."),
        FunctionTool.from_defaults(fn=unit_converter, name="UnitConverter", description="Convert between different units (length, weight, temperature, volume)."),
        FunctionTool.from_defaults(fn=get_current_time, name="TimeChecker", description="Get current date and time."),
        FunctionTool.from_defaults(fn=google_search, name="WebSearch", description="Search the web for current information."),
        FunctionTool.from_defaults(fn=summarize_text, name="TextSummarizer", description="Summarize long text into key points.")
    ]
    
    # Add document query tool if available
    if _document_query_engine:
        document_tool = QueryEngineTool.from_defaults(
            query_engine=_document_query_engine,
            name="DocumentSearch",
            description="Search and answer questions based on uploaded documents."
        )
        tools.append(document_tool)
    
    # Create agent
    agent = ReActAgent.from_tools(
        tools,
        llm=settings.llm,
        memory=settings.memory if hasattr(settings, 'memory') else None,
        verbose=True
    )
    
    return agent

# Streamlit UI
def main():
    # Main header with improved gradient styling
    st.markdown('<h1 class="main-title">Personal AI Assistant ✨</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Your friendly AI companion powered by Llama 3.3</p>', unsafe_allow_html=True)
    
    # Header info cards with high contrast
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('''
        <div class="info-card">
            <h4>Model: Llama 3.3</h4>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        chat_count = len(st.session_state.get('messages', []))
        st.markdown(f'''
        <div class="info-card">
            <h4>Chats: {chat_count}</h4>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        st.markdown('''
        <div class="info-card">
            <h4>Date: Saturday, Jul 05</h4>
        </div>
        ''', unsafe_allow_html=True)
    
    # Sidebar for configuration and document upload
    with st.sidebar:
        st.markdown('<div style="text-align: center; padding: 1rem;"><h2>✨ AI Assistant</h2></div>', unsafe_allow_html=True)
        
        # Quick action buttons
        st.markdown('<div class="capability-card"><h4>🎯 Try asking me...</h4></div>', unsafe_allow_html=True)
        
        if st.button("Tell me a fun fact about space", key="space_fact"):
            st.session_state.sample_question = "Tell me a fun fact about space"
            
        if st.button("What should I cook for dinner?", key="dinner_idea"):
            st.session_state.sample_question = "What should I cook for dinner?"
            
        if st.button("Help me plan my weekend", key="weekend_plan"):
            st.session_state.sample_question = "Help me plan my weekend"
            
        if st.button("Recommend a book to read", key="book_rec"):
            st.session_state.sample_question = "Recommend a book to read"
        
        st.markdown("---")
        
        # My Capabilities section with vibrant styling
        st.markdown('<div class="capability-card"><h4 style="color: #2d3436;">✨ My Capabilities</h4></div>', unsafe_allow_html=True)
        
        # Capability grid with joyful colors
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('''
            <div class="capability-grid-item">
                <div style="font-size: 2rem;">🧠</div>
                <div>Answer Questions</div>
            </div>
            ''', unsafe_allow_html=True)
            
        with col2:
            st.markdown('''
            <div class="capability-grid-item">
                <div style="font-size: 2rem;">📝</div>
                <div>Write Content</div>
            </div>
            ''', unsafe_allow_html=True)
            
        col3, col4 = st.columns(2)
        with col3:
            st.markdown('''
            <div class="capability-grid-item">
                <div style="font-size: 2rem;">💡</div>
                <div>Creative Ideas</div>
            </div>
            ''', unsafe_allow_html=True)
            
        with col4:
            st.markdown('''
            <div class="capability-grid-item">
                <div style="font-size: 2rem;">📊</div>
                <div>Summarize Text</div>
            </div>
            ''', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # User ID for memory
        st.markdown('<div style="color: white; font-weight: 600;">🆔 User ID (for memory)</div>', unsafe_allow_html=True)
        user_id = st.text_input("", value="default_user", label_visibility="collapsed")
        if user_id != st.session_state.get('user_id', 'default_user'):
            st.session_state.user_id = user_id
            st.cache_resource.clear()
        
        st.markdown("---")
        
        # Document upload section
        st.markdown('<div style="color: white; font-weight: 600; font-size: 1.2rem;">📄 Document Upload</div>', unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "",
            accept_multiple_files=True,
            type=['txt', 'pdf', 'md', 'csv', 'json'],
            help="Upload documents to analyze",
            label_visibility="collapsed"
        )
        
        document_query_engine = None
        if uploaded_files:
            with st.spinner("Processing documents..."):
                documents = process_uploaded_files(uploaded_files)
                if documents:
                    document_query_engine = create_document_index(documents)
                    st.success(f"✅ Processed {len(documents)} documents")
                    
                    # Show document info
                    st.markdown('<div style="color: white; font-weight: 600;">**📋 Uploaded Documents:**</div>', unsafe_allow_html=True)
                    for doc in documents:
                        filename = doc.metadata.get('filename', 'Unknown')
                        st.markdown(f'<div style="color: #ffeaa7;">📄 {filename}</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Available tools with joyful styling
        st.markdown('<div style="color: white; font-weight: 600; font-size: 1.2rem;">🛠️ Available Tools</div>', unsafe_allow_html=True)
        st.markdown('''
        <div class="tools-list">
        <div style="color: #2d3436; font-weight: 600;">
        🧮 Calculator & Math<br>
        📝 Text Analysis<br>
        💱 Currency & Units<br>
        🔍 Web Search<br>
        📊 Document Analysis<br>
        ⏰ Time & Date<br>
        📋 Text Summarization
        </div>
        </div>
        ''', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Sample questions with vibrant buttons
        st.markdown('<div style="color: white; font-weight: 600; font-size: 1.2rem;">💡 Sample Questions</div>', unsafe_allow_html=True)
        sample_questions = [
            "What's 15% of 250?",
            "Convert 100 USD to EUR",
            "How many words are in this text?",
            "Convert 100 km to miles",
            "What time is it now?",
            "Search for latest AI news",
            "Summarize this document",
            "What's the weather like?"
        ]
        
        for question in sample_questions:
            if st.button(question, key=f"sample_{question}"):
                st.session_state.sample_question = question
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Initialize or update agent with document query engine
    if document_query_engine:
        st.session_state.agent = initialize_agent(document_query_engine)
    elif "agent" not in st.session_state:
        with st.spinner("🚀 Initializing AI Assistant..."):
            st.session_state.agent = initialize_agent()
    
    # Display chat messagesa
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Handle sample question
    if hasattr(st.session_state, 'sample_question'):
        prompt = st.session_state.sample_question
        del st.session_state.sample_question
    else:
        prompt = st.chat_input("Ask me anything...")
    
    if prompt:
        # Display user message
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Get response from agent
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.agent.chat(prompt)
                    st.markdown(str(response))
                    st.session_state.messages.append({"role": "assistant", "content": str(response)})
                except Exception as e:
                    error_message = f"I encountered an error: {str(e)}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})

    # Status indicators at bottom
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if get_memory():
            st.markdown('<div class="status-card"><div style="color: #4CAF50;">🧠 Memory Active</div></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-card"><div style="color: #9E9E9E;">💭 Memory Inactive</div></div>', unsafe_allow_html=True)
    
    with col2:
        if uploaded_files:
            st.markdown(f'<div class="status-card"><div style="color: #4CAF50;">📄 {len(uploaded_files)} Documents</div></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-card"><div style="color: #9E9E9E;">📄 No Documents</div></div>', unsafe_allow_html=True)
    
    with col3:
        if env_vars["GOOGLE_API_KEY"]:
            st.markdown('<div class="status-card"><div style="color: #4CAF50;">🔍 Web Search Active</div></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-card"><div style="color: #9E9E9E;">🔍 Web Search Inactive</div></div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()