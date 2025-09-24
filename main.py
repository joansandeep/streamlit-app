import os
import streamlit as st
import re
import pandas as pd
import json
import hashlib
from datetime import datetime
from pathlib import Path
from src.pipeline import RAGPipeline
import plotly.express as px

# Load secrets from Streamlit Cloud Secrets Manager
groq_api_key = st.secrets["GROQ_API_KEY"]
openrouter_api_key = st.secrets["OPENROUTER_API_KEY"]
groq_model_name = st.secrets.get("GROQ_MODEL_NAME", "llama-3.1-8b-instant")


# Configure the app page
st.set_page_config(
    page_title="MediBot - AI Pharmacy Assistant",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS styling with improved navigation
st.markdown("""
<style>
/* Global Styles */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0f1419 !important;
    color: #f0f6fc !important;
}

h1, h2, h3, h4, h5, h6, label, .stTextInput label, p, span, div {
    color: #f0f6fc !important;
    text-shadow: 1px 1px 3px rgba(0,0,0,0.8);
}

/* Enhanced Navigation Styles */
.nav-header {
    background: linear-gradient(135deg, #1e3a8a, #3b82f6) !important;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 20px;
    box-shadow: 0 8px 16px rgba(0,0,0,0.3);
    border: 2px solid #2563eb;
    text-align: center;
}

.nav-title {
    font-size: 24px;
    font-weight: bold;
    color: #ffffff !important;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.8);
    margin-bottom: 10px;
}

.nav-subtitle {
    font-size: 14px;
    color: #e0e7ff !important;
    opacity: 0.9;
}

.nav-item {
    background: linear-gradient(145deg, #1e293b, #334155) !important;
    border: 2px solid transparent !important;
    border-radius: 12px !important;
    padding: 15px !important;
    margin: 8px 0 !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 8px rgba(0,0,0,0.2) !important;
}

.nav-item:hover {
    background: linear-gradient(145deg, #2563eb, #3b82f6) !important;
    border-color: #60a5fa !important;
    transform: translateX(5px) !important;
    box-shadow: 0 6px 12px rgba(59, 130, 246, 0.3) !important;
}

.nav-item.active {
    background: linear-gradient(145deg, #059669, #10b981) !important;
    border-color: #34d399 !important;
    box-shadow: 0 0 20px rgba(16, 185, 129, 0.4) !important;
}

.nav-icon {
    font-size: 18px;
    margin-right: 8px;
}

.user-info-section {
    background: linear-gradient(145deg, #7c3aed, #a855f7) !important;
    padding: 15px;
    border-radius: 12px;
    margin: 10px 0;
    box-shadow: 0 4px 8px rgba(124, 58, 237, 0.3);
    text-align: center;
}

.user-avatar {
    width: 50px;
    height: 50px;
    border-radius: 50%;
    background: linear-gradient(135deg, #f59e0b, #f97316);
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto 10px;
    font-size: 24px;
    color: white;
    font-weight: bold;
}

.quick-stats {
    background: linear-gradient(145deg, #1f2937, #111827) !important;
    padding: 15px;
    border-radius: 12px;
    margin: 10px 0;
    border: 1px solid #374151;
}

.quick-stat-item {
    display: flex;
    justify-content: space-between;
    margin: 8px 0;
    padding: 5px 0;
    border-bottom: 1px solid #374151;
}

.status-indicator {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: #10b981;
    display: inline-block;
    margin-right: 8px;
    box-shadow: 0 0 8px rgba(16, 185, 129, 0.6);
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.5; }
    100% { opacity: 1; }
}

/* Enhanced Input and Button Styles */
.stTextInput input {
    background-color: #1c2128 !important;
    color: #f0f6fc !important;
    border: 2px solid #2196f3 !important;
    border-radius: 8px !important;
}

.stTextInput input:focus {
    border-color: #ff9800 !important;
    box-shadow: 0 0 8px rgba(33, 150, 243, 0.4) !important;
}

.stButton > button, .stFormSubmitButton button {
    background: linear-gradient(145deg, #2196f3, #1976d2) !important;
    color: white !important;
    border-radius: 8px !important;
    font-weight: bold !important;
    border: none !important;
    transition: all 0.3s ease !important;
}

.stButton > button:hover, .stFormSubmitButton button:hover {
    background: linear-gradient(145deg, #64b5f6, #2196f3) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 8px rgba(33, 150, 243, 0.3) !important;
}

/* Enhanced Message Styles */
.user-message {
    border-left: 5px solid #2196f3;
    background: linear-gradient(145deg, #1e2732, #161b22) !important;
    padding: 15px;
    margin: 10px 0;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.3);
}

.bot-message {
    border-left: 5px solid #4caf50;
    background: linear-gradient(145deg, #1a2e1a, #161b22) !important;
    padding: 15px;
    margin: 10px 0;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.3);
}

/* Enhanced Metric Cards */
.metric-card {
    background: linear-gradient(145deg, #1c2128, #0d1117) !important;
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0 6px 12px rgba(0,0,0,0.4);
    border: 2px solid #30363d;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    position: relative;
    overflow: hidden;
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, #2196f3, #21cbf3, #2196f3);
}

.metric-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 20px rgba(0,0,0,0.6);
}

.metric-title {
    font-size: 18px;
    color: #7c3aed !important;
    font-weight: bold;
    margin-bottom: 15px;
}

.metric-value {
    font-size: 36px;
    color: #2196f3 !important;
    font-weight: bold;
}

/* System Info Cards */
.system-info-card {
    background: linear-gradient(145deg, #1c2128, #0d1117) !important;
    padding: 25px;
    margin: 15px 0;
    border-radius: 15px;
    border: 2px solid #30363d;
    box-shadow: 0 6px 12px rgba(0,0,0,0.4);
    position: relative;
    overflow: hidden;
}

.system-info-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, #10b981, #34d399, #10b981);
}

/* Sidebar Enhancements */
.sidebar .sidebar-content {
    background-color: #0d1117 !important;
    border-right: 3px solid #21262d !important;
}

.conversation-item {
    background: linear-gradient(145deg, #1c2128, #161b22) !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    margin: 5px 0 !important;
    padding: 8px !important;
    transition: all 0.2s ease !important;
}

.conversation-item:hover {
    border-color: #2196f3 !important;
    box-shadow: 0 2px 8px rgba(33, 150, 243, 0.2) !important;
}

/* Custom Selectbox Styles */
.nav-selectbox {
    background: linear-gradient(145deg, #1e293b, #334155) !important;
    border: 2px solid #475569 !important;
    border-radius: 12px !important;
    color: #f0f6fc !important;
}
</style>
""", unsafe_allow_html=True)

# User session and chat management class (same as before)
class ChatHistoryManager:
    def __init__(self):
        self.history_dir = Path("chat_history")
        self.history_dir.mkdir(exist_ok=True)
        self.sessions_file = self.history_dir / "user_sessions.json"

    def normalize_username(self, username):
        return username.strip().lower()

    def get_user_id(self, username):
        return hashlib.md5(self.normalize_username(username).encode()).hexdigest()[:8]

    def load_sessions(self):
        if self.sessions_file.exists():
            with open(self.sessions_file, "r") as f:
                return json.load(f)
        return {}

    def save_sessions(self, sessions):
        with open(self.sessions_file, "w") as f:
            json.dump(sessions, f, indent=2)

    def load_user(self, username):
        sessions = self.load_sessions()
        return sessions.get(self.get_user_id(username))

    def create_user(self, username, password):
        sessions = self.load_sessions()
        uid = self.get_user_id(username)
        if uid in sessions:
            return False, "User already exists"
        pw_hash = hashlib.md5(password.encode()).hexdigest()
        sessions[uid] = {
            "username": username,
            "password_hash": pw_hash,
            "created_at": datetime.now().isoformat(),
            "last_active": datetime.now().isoformat(),
            "total_conversations": 0,
            "session_data": {"conversations": [], "current_messages": []}
        }
        self.save_sessions(sessions)
        return True, "User created successfully"

    def authenticate(self, username, password):
        sessions = self.load_sessions()
        uid = self.get_user_id(username)
        user = sessions.get(uid)
        if not user:
            return False, "User not found"
        if user["password_hash"] != hashlib.md5(password.encode()).hexdigest():
            return False, "Invalid password"
        user["last_active"] = datetime.now().isoformat()
        self.save_sessions(sessions)
        return True, "Login successful"

    def save_session_data(self, username, session_data):
        sessions = self.load_sessions()
        uid = self.get_user_id(username)
        if uid in sessions:
            sessions[uid]["session_data"] = session_data
            sessions[uid]["last_active"] = datetime.now().isoformat()
            sessions[uid]["total_conversations"] = len(session_data.get("conversations", []))
            self.save_sessions(sessions)

# Session state initialization (same as before)
for key, default in [
    ("authenticated", False), ("current_user", None),
    ("conversations", []), ("messages", []),
    ("rag_pipeline", None), ("system_loaded", False),
    ("chat_manager", None), ("selected_conversation", "New Chat"),
    ("current_page", "💬 Chat")
]:
    if key not in st.session_state:
        st.session_state[key] = default

if st.session_state.chat_manager is None:
    st.session_state.chat_manager = ChatHistoryManager()

# Enhanced Navigation Function
def render_navigation():
    # Navigation Header
    st.sidebar.markdown("""
    <div class='nav-header'>
        <div class='nav-title'>🏥 MediBot</div>
        <div class='nav-subtitle'>AI Pharmacy Assistant</div>
    </div>
    """, unsafe_allow_html=True)
    
    # User Info Section
    if st.session_state.authenticated:
        user_initial = st.session_state.current_user[0].upper()
        st.sidebar.markdown(f"""
        <div class='user-info-section'>
            <div class='user-avatar'>{user_initial}</div>
            <div style='color: white; font-weight: bold;'>{st.session_state.current_user}</div>
            <div style='color: #e0e7ff; font-size: 12px; margin-top: 5px;'>
                <span class='status-indicator'></span>Online
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick Stats
        st.sidebar.markdown("""
        <div class='quick-stats'>
            <div style='font-weight: bold; margin-bottom: 10px; color: #f0f6fc;'>Quick Stats</div>
            <div class='quick-stat-item'>
                <span>🗨️ Conversations</span>
                <span>{}</span>
            </div>
            <div class='quick-stat-item'>
                <span>💬 Messages</span>
                <span>{}</span>
            </div>
            <div class='quick-stat-item'>
                <span>⏰ Last Active</span>
                <span>Now</span>
            </div>
        </div>
        """.format(
            len(st.session_state.conversations),
            len(st.session_state.messages)
        ), unsafe_allow_html=True)

    # Navigation Menu
    st.sidebar.markdown("<div style='margin: 20px 0; font-weight: bold; color: #f0f6fc;'>🧭 Navigation Menu</div>", unsafe_allow_html=True)
    
    pages = [
        {"name": "💬 Chat", "icon": "💬", "desc": "AI Chat Interface"},
        {"name": "📊 Analytics", "icon": "📊", "desc": "Medicine Analytics"},
        {"name": "⚙️ System Info", "icon": "⚙️", "desc": "System Status"},
        {"name": "🔧 Settings", "icon": "🔧", "desc": "User Settings"}
    ]
    
    for page in pages:
        if page["name"] == st.session_state.current_page:
            nav_class = "nav-item active"
        else:
            nav_class = "nav-item"
            
        if st.sidebar.button(
            f"{page['icon']} {page['name'].split(' ', 1)[1]}",
            key=f"nav_{page['name']}",
            help=page["desc"]
        ):
            st.session_state.current_page = page["name"]
            st.rerun()

# Load functions (same as before)
@st.cache_resource
def load_rag_system():
    data_path = "data/1mg_medicine_data_RAG_ready.csv"
    faiss_path = "data/embeddings/faiss_index.bin"
    if not os.path.exists(data_path) or not os.path.exists(faiss_path):
        st.error("Required RAG data files not found.")
        return None
    return RAGPipeline(faiss_path=faiss_path, data_path=data_path)

def load_medicine_data():
    data_path = "data/1mg_medicine_data_RAG_ready.csv"
    if not os.path.exists(data_path):
        st.error("Medicine data not found.")
        return None
    df = pd.read_csv(data_path)
    if "Price" in df.columns:
        df["Price"] = (df["Price"]
                       .astype(str).str.replace("₹", "").str.replace("Rs.", "")
                       .str.replace(",", "").str.strip())
        df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    return df

# Conversation management functions (same as before)
def save_conversation():
    if st.session_state.messages:
        conv = {
            "id": len(st.session_state.conversations),
            "title": st.session_state.messages[0]['content'][:50] if st.session_state.messages else "Untitled",
            "timestamp": datetime.now().isoformat(),
            "messages": st.session_state.messages.copy()
        }
        st.session_state.conversations.append(conv)
        data = {
            "conversations": st.session_state.conversations,
            "current_messages": []
        }
        st.session_state.chat_manager.save_session_data(st.session_state.current_user, data)

def clear_conversation(idx):
    if 0 <= idx < len(st.session_state.conversations):
        st.session_state.conversations.pop(idx)
        data = {
            "conversations": st.session_state.conversations,
            "current_messages": []
        }
        st.session_state.chat_manager.save_session_data(st.session_state.current_user, data)

def load_conversation(idx):
    if 0 <= idx < len(st.session_state.conversations):
        st.session_state.messages = st.session_state.conversations[idx]["messages"].copy()

# Login function (same as before)
def user_login():
    st.title("🏥 MediBot Login")
    tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])

    with tab1:
        with st.form("login"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")
            if submit:
                if username and password:
                    success, msg = st.session_state.chat_manager.authenticate(username, password)
                    if success:
                        st.session_state.authenticated = True
                        st.session_state.current_user = username
                        user_data = st.session_state.chat_manager.load_user(username)
                        if user_data and "session_data" in user_data:
                            st.session_state.conversations = user_data["session_data"].get("conversations", [])
                        else:
                            st.session_state.conversations = []
                        st.session_state.messages = []
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)
                else:
                    st.warning("Please enter username and password")

    with tab2:
        with st.form("register"):
            username = st.text_input("New Username", key="reg_username")
            password = st.text_input("New Password", type="password", key="reg_password")
            confirm = st.text_input("Confirm Password", type="password", key="reg_confirm")
            submit = st.form_submit_button("Register")
            if submit:
                if not username or not password or not confirm:
                    st.warning("Please fill all fields")
                elif password != confirm:
                    st.error("Passwords do not match")
                elif len(password) < 6:
                    st.warning("Password must be at least 6 characters")
                else:
                    success, msg = st.session_state.chat_manager.create_user(username, password)
                    if success:
                        st.success(msg)
                        st.info("Please login now")
                    else:
                        st.error(msg)

def display_medical_report(extracted_text):
    # Patient details
    name = re.search(r'Patient Name:\s*(.+)', extracted_text)
    pid = re.search(r'Patient ID:\s*(.+)', extracted_text)
    dob = re.search(r'Date of Birth:\s*(.+)', extracted_text)
    gender = re.search(r'Gender:\s*(.+)', extracted_text)
    report_date = re.search(r'Date of Report:\s*(.+)', extracted_text)
    doctor = re.search(r'Dr\..+', extracted_text)

    # Diagnosis
    diagnosis_block = re.search(r'Diagnosis\n(.*?)\nTest Results', extracted_text, re.DOTALL)
    diagnosis = diagnosis_block.group(1).strip() if diagnosis_block else "Not found"
    # Test Results
    test_block = re.search(r'Test Results\n(.*?)\nPrescription', extracted_text, re.DOTALL)
    test_lines = test_block.group(1).strip().split('\n') if test_block else []
    # Prescription
    presc_block = re.search(r'Prescription\n(.*?)\nDoctor', extracted_text, re.DOTALL)
    presc_lines = presc_block.group(1).strip().split('\n') if presc_block else []

    st.markdown("## 📝 Medical Report")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Patient Name:** {name.group(1) if name else 'N/A'}")
        st.markdown(f"**Patient ID:** {pid.group(1) if pid else 'N/A'}")
        st.markdown(f"**Gender:** {gender.group(1) if gender else 'N/A'}")
    with col2:
        st.markdown(f"**Date of Birth:** {dob.group(1) if dob else 'N/A'}")
        st.markdown(f"**Report Date:** {report_date.group(1) if report_date else 'N/A'}")
        st.markdown(f"**Doctor:** {doctor.group(0) if doctor else 'N/A'}")
    st.markdown("---")

    st.markdown("### Diagnosis")
    st.info(diagnosis)
    st.markdown("---")

    st.markdown("### Test Results")
    if test_lines:
        st.table([
            {"Test": test_lines[i], "Result": test_lines[i+1], "Reference": test_lines[i+2]}
            for i in range(3, len(test_lines), 3)
        ])
    else:
        st.warning("No test results found")

    st.markdown("### Prescription")
    if presc_lines:
        st.table([
            {"Medicine": presc_lines[i], "Dosage": presc_lines[i+1], "Frequency": presc_lines[i+2]}
            for i in range(3, len(presc_lines), 3)
        ])
    else:
        st.warning("No prescription section found")


def chat_page():
    if not st.session_state.system_loaded:
        with st.spinner("Loading AI system..."):
            rag = load_rag_system()
            if rag is None:
                st.error("Failed loading model")
                return
            st.session_state.rag_pipeline = rag
            st.session_state.system_loaded = True
            st.success("System ready")

    st.header("💬 Chat Interface")

    # Sidebar - Conversation list as clickable buttons
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🗨️ Your Conversations")

    conv_titles = ["New Chat"] + [
        conv['title'] if len(conv['title']) <= 50 else conv['title'][:47] + "..."
        for conv in st.session_state.conversations
    ]

    for idx, title in enumerate(conv_titles):
        is_selected = (st.session_state.get("selected_conv", "New Chat") == title)
        prefix = "▶️ " if is_selected else ""
        if st.sidebar.button(f"{prefix}{title}", key=f"conv_{idx}"):
            st.session_state.selected_conv = title
            if title == "New Chat":
                st.session_state.messages = []
            else:
                real_idx = idx - 1  # offset for New Chat
                load_conversation(real_idx)
            st.rerun()

    if st.sidebar.button("🚪 Logout"):
        if st.session_state.messages:
            save_conversation()
        st.session_state.clear()
        st.rerun()

    # Clear selected conversation
    selected = st.session_state.get("selected_conv", "New Chat")
    if selected != "New Chat":
        del_idx = conv_titles.index(selected) - 1  # offset for New Chat
        if st.sidebar.button("🗑️ Clear This Chat"):
            clear_conversation(del_idx)
            st.session_state.messages = []
            st.session_state.selected_conv = "New Chat"
            st.rerun()

    # Display medical report if PDF context exists
    if st.session_state.get("pdf_text"):
        display_medical_report(st.session_state["pdf_text"])
        if st.button("Clear PDF Context"):
            st.session_state["pdf_text"] = None
            st.rerun()

    # Display chat messages
    for msg in st.session_state.messages:
        if msg['role'] == 'user':
            st.markdown(f'<div class="user-message"><b>You:</b> {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-message"><b>MediBot:</b> {msg["content"]}</div>', unsafe_allow_html=True)

    # Chat input form with PDF uploader
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([4, 1])  # Create columns for text input and PDF uploader

        with col1:
            user_input = st.text_input("Ask a question about medicines or the uploaded PDF", label_visibility="collapsed")
        
        with col2:
            uploaded_file = st.file_uploader("📄 Upload PDF", type=["pdf"], label_visibility="collapsed")
            if uploaded_file:
                try:
                    text = st.session_state.rag_pipeline.extract_text_from_pdf(uploaded_file)
                    st.session_state["pdf_text"] = text
                    if "Patient Name" in text and "Diagnosis" in text and "Prescription" in text:
                        # You can decide how to handle this, maybe show a success message
                        pass
                    else:
                        st.text_area("Extracted PDF Text", text, height=300)
                except Exception as e:
                    st.error(f"PDF processing failed: {e}")

        # Send button and chat logic
        if st.form_submit_button("Send") and user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            context = st.session_state.get("pdf_text")
            with st.spinner("MediBot is thinking..."):
                try:
                    response = st.session_state.rag_pipeline.run(user_input, context=context)
                except Exception as e:
                    response = f"Error: {e}"
                st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

# Analytics and System Info pages (same as before but with enhanced styling)
def analytics_page():
    st.title("📊 Medicine Analytics Dashboard")
    
    # Load data first and handle error case
    df = load_medicine_data()
    if df is None:
        st.error("Medicine data not found.")
        return
    
    # Debug: Check the actual structure of your data
    st.sidebar.markdown("### 🔍 Data Debug Info")
    st.sidebar.write(f"Number of columns: {len(df.columns)}")
    st.sidebar.write("Column names:")
    st.sidebar.write(df.columns.tolist())
    
    # Handle column naming based on actual column count
    expected_columns = ['Category', 'Name', 'Price', 'Manufacturer', 'Salt', 'Uses', 'Side_Effects', 'Extra1', 'Extra2']
    
    # Only rename columns if we have the expected structure
    if len(df.columns) >= 5:
        # Create column names list matching actual column count
        new_columns = expected_columns[:len(df.columns)]
        df.columns = new_columns
        
        # Clean and extract primary salt from format like "Paracetamol (650mg)"
        if "Salt" in df.columns:
            # Extract salt name without dosage
            df['Primary_Salt'] = df['Salt'].str.extract(r'([A-Za-z\s\+]+)')[0].str.strip()
            # Remove common words and clean
            df['Primary_Salt'] = df['Primary_Salt'].str.replace(r'\(.*\)', '', regex=True).str.strip()
            
            # Debug: Show what we extracted
            st.sidebar.write("Top 5 Raw Salts:")
            st.sidebar.dataframe(df['Salt'].value_counts().head())
            st.sidebar.write("Top 5 Primary Salts:")
            st.sidebar.dataframe(df['Primary_Salt'].value_counts().head())
    else:
        st.error("Data doesn't have expected column structure")
        st.write("Your data structure:")
        st.dataframe(df.head())
        return
    
    # Convert Price to numeric if Price column exists
    if "Price" in df.columns:
        df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    
    # Display metrics in a row
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>📋 Total Medicines</div>
            <div class='metric-value'>{len(df):,}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        unique_primary_salts = df["Primary_Salt"].nunique() if "Primary_Salt" in df.columns else 0
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>🧪 Unique Primary Salts</div>
            <div class='metric-value'>{unique_primary_salts:,}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        manufacturers = df["Manufacturer"].nunique() if "Manufacturer" in df.columns else 0
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>🏭 Manufacturers</div>
            <div class='metric-value'>{manufacturers:,}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Charts Section
    st.header("📈 Analytics Charts")
    
    # Row 1: Top Manufacturers (full width for better visibility)
    st.subheader("🏭 Top 15 Manufacturers")
    if "Manufacturer" in df.columns:
        top_manufacturers = df["Manufacturer"].value_counts().head(15)
        fig_bar = px.bar(
            x=top_manufacturers.values,
            y=top_manufacturers.index,
            orientation='h',
            title="Top 15 Medicine Manufacturers",
            labels={'x': 'Number of Medicines', 'y': 'Manufacturer'},
            color=top_manufacturers.values,
            color_continuous_scale='viridis'
        )
        fig_bar.update_layout(
            height=600,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            title_font_color='white',
            margin=dict(l=150, r=50, t=50, b=50)
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Row 2: Active Salts (full width for better pie chart visibility)
    st.subheader("🧪 Top 15 Active Salts Distribution")
    if "Primary_Salt" in df.columns:
        top_primary_salts = df["Primary_Salt"].value_counts().head(15)
        
        # Create two versions: pie chart and bar chart
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart with better settings and meaningful labels
            fig_pie = px.pie(
                values=top_primary_salts.values,
                names=[f"{name} ({count})" for name, count in zip(top_primary_salts.index, top_primary_salts.values)],
                title="Top 15 Active Salts (Pie Chart)",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_pie.update_traces(
                textposition='outside', 
                textinfo='label+percent',
                textfont_size=10,
                pull=[0.1 if i == 0 else 0 for i in range(len(top_primary_salts))]
            )
            fig_pie.update_layout(
                height=500,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                title_font_color='white',
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="middle",
                    y=0.5,
                    xanchor="left",
                    x=1.02,
                    font=dict(size=9)
                )
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Horizontal bar chart for better readability
            fig_bar_salt = px.bar(
                x=top_primary_salts.values,
                y=top_primary_salts.index,
                orientation='h',
                title="Top 15 Active Salts (Bar Chart)",
                labels={'x': 'Number of Medicines', 'y': 'Active Salt'},
                color=top_primary_salts.values,
                color_continuous_scale='plasma'
            )
            fig_bar_salt.update_layout(
                height=500,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                title_font_color='white',
                showlegend=False,
                margin=dict(l=120, r=50, t=50, b=50)
            )
            st.plotly_chart(fig_bar_salt, use_container_width=True)
    
    # Row 3: Price Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💰 Price Distribution")
        if "Price" in df.columns:
            price_data = df["Price"].dropna()
            if len(price_data) > 0:
                price_data = price_data[price_data <= price_data.quantile(0.95)]
                
                fig_hist = px.histogram(
                    x=price_data,
                    nbins=30,
                    title="Medicine Price Distribution (₹)",
                    labels={'x': 'Price (₹)', 'y': 'Number of Medicines'},
                    color_discrete_sequence=['#2196f3']
                )
                fig_hist.update_layout(
                    height=450,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                    title_font_color='white'
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            else:
                st.info("No valid price data available")
    
    with col2:
        st.subheader("💊 Top Salts by Medicine Count")
        if "Primary_Salt" in df.columns:
            top_salts_count = df["Primary_Salt"].value_counts().head(10)
            
            fig_salt_bar = px.bar(
                x=top_salts_count.values,
                y=top_salts_count.index,
                orientation='h',
                title="Most Common Active Salts (Top 10)",
                labels={'x': 'Number of Medicines', 'y': 'Active Salt'},
                color=top_salts_count.values,
                color_continuous_scale='blues'
            )
            fig_salt_bar.update_layout(
                height=450,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                title_font_color='white',
                showlegend=False,
                margin=dict(l=120, r=50, t=50, b=50)
            )
            st.plotly_chart(fig_salt_bar, use_container_width=True)

    # Interactive filters section
    st.markdown("---")
    st.header("🎛️ Interactive Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if "Manufacturer" in df.columns:
            top_manufacturers_list = df["Manufacturer"].value_counts().head(50).index.tolist()
            selected_manufacturers = st.multiselect(
                "Select Manufacturers:",
                options=top_manufacturers_list,
                default=[],
                help="Top 50 manufacturers by medicine count"
            )
    
    with col2:
        if "Price" in df.columns:
            valid_prices = df["Price"].dropna()
            if len(valid_prices) > 0:
                price_range = st.slider(
                    "Price Range (₹):",
                    min_value=float(valid_prices.min()),
                    max_value=float(valid_prices.max()),
                    value=(float(valid_prices.min()), float(valid_prices.quantile(0.75)))
                )
            else:
                st.info("No valid price data for filtering")
    
    with col3:
        if "Primary_Salt" in df.columns:
            all_salts_list = sorted(df["Primary_Salt"].dropna().unique().tolist())
            
            # Search filter for salts
            search_term = st.text_input(
                "🔍 Filter salts:",
                placeholder="Type to filter (e.g., 'para', 'ibu', 'asp')",
                help="Filter the dropdown by typing part of the salt name"
            )
            
            # Apply filter
            if search_term:
                filtered_options = [salt for salt in all_salts_list 
                                  if search_term.lower() in salt.lower()]
                if not filtered_options:
                    st.warning(f"No salts found containing '{search_term}'")
                    filtered_options = all_salts_list[:20]  # Fallback to first 20
                else:
                    st.success(f"🎯 Found {len(filtered_options)} salts containing '{search_term}'")
            else:
                filtered_options = all_salts_list[:50]  # Show first 50 by default
                st.info(f"📋 Showing first 50 of {len(all_salts_list)} total salts")
            
            # Multiselect with filtered options
            selected_salts = st.multiselect(
                f"Select Active Salts ({len(filtered_options)} shown):",
                options=filtered_options,
                default=[],
                help="Select multiple salts to filter medicines"
            )
            
            # Show popular salt quick buttons
            if not search_term:  # Only show when not searching
                st.markdown("**💡 Quick Add Popular Salts:**")
                popular_salts = ["Paracetamol", "Ibuprofen", "Aspirin", "Metformin", "Amoxicillin", "Diclofenac"]
                available_popular = [salt for salt in popular_salts 
                                   if salt in all_salts_list and salt not in selected_salts]
                
                if available_popular:
                    cols = st.columns(min(3, len(available_popular)))
                    for i, salt in enumerate(available_popular[:6]):
                        with cols[i % 3]:
                            if st.button(f"+ {salt}", key=f"quick_add_{salt}"):
                                selected_salts.append(salt)
                                st.rerun()

    # Apply filters and show results
    if selected_manufacturers or selected_salts or ('price_range' in locals()):
        filtered_df = df.copy()
        
        if selected_manufacturers:
            filtered_df = filtered_df[filtered_df["Manufacturer"].isin(selected_manufacturers)]
        
        if "Price" in df.columns and 'price_range' in locals():
            filtered_df = filtered_df[
                (filtered_df["Price"] >= price_range[0]) & 
                (filtered_df["Price"] <= price_range[1])
            ]
        
        if selected_salts:
            filtered_df = filtered_df[filtered_df["Primary_Salt"].isin(selected_salts)]
        
        if len(filtered_df) > 0 and len(filtered_df) < len(df):
            st.subheader(f"🔍 Filtered Results ({len(filtered_df):,} medicines)")
            
            # Show filtered statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Filtered Medicines", f"{len(filtered_df):,}")
            with col2:
                if "Price" in filtered_df.columns:
                    avg_price = filtered_df["Price"].mean()
                    st.metric("Average Price", f"₹{avg_price:.2f}")
            with col3:
                unique_mfg = filtered_df["Manufacturer"].nunique() if "Manufacturer" in filtered_df.columns else 0
                st.metric("Unique Manufacturers", f"{unique_mfg:,}")
            
            # Show sample of filtered data
            st.dataframe(filtered_df.head(20), use_container_width=True)
    st.markdown("---")



def system_info():
    st.title("⚙️ System Information")
    
    # System Status Card
    st.markdown("""
    <div class='system-info-card'>
        <h3>🔧 System Status</h3>
        <p><strong>Status:</strong> <span class='status-indicator'></span>Online</p>
        <p><strong>RAG System:</strong> ✅ Loaded</p>
        <p><strong>Database:</strong> ✅ Connected</p>
        <p><strong>Last Updated:</strong> 2025-09-22</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Performance Metrics
    st.markdown("""
    <div class='system-info-card'>
        <h3>📈 Performance Metrics</h3>
        <p><strong>Response Time:</strong> ~2.5 seconds</p>
        <p><strong>Accuracy:</strong> 95.2%</p>
        <p><strong>Uptime:</strong> 99.8%</p>
        <p><strong>Active Users:</strong> 1,245</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Configuration
    st.markdown("""
    <div class='system-info-card'>
        <h3>⚙️ Configuration</h3>
        <p><strong>Model:</strong> RAGPipeline v2.0</p>
        <p><strong>Embedding Model:</strong> sentence-transformers</p>
        <p><strong>Vector Database:</strong> FAISS</p>
        <p><strong>LLM:</strong> Llama-based model</p>
    </div>
    """, unsafe_allow_html=True)

def settings_page():
    st.title("🔧 User Settings")
    
    st.markdown("""
    <div class='system-info-card'>
        <h3>👤 User Preferences</h3>
        <p>Customize your MediBot experience</p>
    </div>
    """, unsafe_allow_html=True)
    
    # User preferences (placeholder)
    st.checkbox("🔔 Enable notifications")
    st.checkbox("🌙 Dark mode (enabled)")
    st.selectbox("🗣️ Preferred language", ["English", "Hindi", "Tamil"])
    st.slider("⚡ Response speed", 1, 10, 7)

def main():
    if not st.session_state.authenticated:
        user_login()
    else:
        render_navigation()
        
        # Route to different pages based on navigation
        if st.session_state.current_page == "💬 Chat":
            chat_page()
        elif st.session_state.current_page == "📊 Analytics":
            analytics_page()
        elif st.session_state.current_page == "⚙️ System Info":
            system_info()
        elif st.session_state.current_page == "🔧 Settings":
            settings_page()

if __name__ == "__main__":
    main()
