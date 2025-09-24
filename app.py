import streamlit as st
import sys, os
import json
import spacy
import nltk
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
from collections import Counter
import re
from textstat import flesch_reading_ease
import numpy as np
import pandas as pd 
import torch
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import tempfile

# Offline TTS (recommended to avoid SSL/network issues)
# pip install pyttsx3
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False

# -----------------------
# Session state defaults
# -----------------------
if "questions" not in st.session_state:
    st.session_state.questions = []
if "displayed_text" not in st.session_state:
    st.session_state.displayed_text = ""
if "article_title" not in st.session_state:
    st.session_state.article_title = ""

# -----------------------
# NLTK data
# -----------------------
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')

# -----------------------
# spaCy model
# -----------------------
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.error("Please install spaCy English model: python -m spacy download en_core_web_sm")

# -----------------------
# Local imports
# -----------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from NER.extractor import fetch_article_text, extract_named_entities

# -----------------------
# Models
# -----------------------
@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t5_model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)
    t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
    try:
        question_generator = pipeline(
            "text2text-generation",
            model="valhalla/t5-small-qg-hl",
            tokenizer="valhalla/t5-small-qg-hl"
        )
    except Exception:
        question_generator = None
    return t5_model, t5_tokenizer, question_generator, device

class SmartQuestionFilter:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.question_starters = {
            'what','who','when','where','why','how','which','whose','whom',
            'did','do','does','is','are','was','were','will','would','could',
            'should','can','may'
        }
    def is_valid_question_format(self, question):
        question = question.strip()
        if not question.endswith('?'):
            return False
        first_word = question.split()[0].lower()
        if first_word not in self.question_starters:
            return False
        wc = len(question.split())
        return 3 <= wc <= 25
    def has_sufficient_content(self, question, content):
        q_words = set(word_tokenize(question.lower()))
        c_words = set(word_tokenize(content.lower()))
        q_words = {w for w in q_words if w not in self.stop_words and w not in string.punctuation}
        c_words = {w for w in c_words if w not in self.stop_words and w not in string.punctuation}
        return len(q_words.intersection(c_words)) >= 2
    def is_meaningful_question(self, question, entities):
        ql = question.lower()
        generic_patterns = [
            r"^what is \w+\?$",
            r"^who is \w+\?$",
            r"^when did \w+\?$",
            r"^what happened in \w+\?$"
        ]
        for pattern in generic_patterns:
            if re.match(pattern, ql):
                entity_in_q = re.findall(r'\b\w+\b', ql)[-2]
                entity_mentions = [ent[0].lower() for ent in entities if entity_in_q in ent[0].lower()]
                if len(entity_mentions) == 0:
                    return False
        return True
    def remove_duplicate_questions(self, questions):
        unique = []
        seen = set()
        for q in questions:
            pattern = re.sub(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', 'ENTITY', q)
            pattern = re.sub(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b', 'NUMBER', pattern)
            if pattern not in seen:
                seen.add(pattern)
                unique.append(q)
        return unique
    def score_question_quality(self, question, content, entities):
        score = 0
        ql = question.lower()
        wc = len(question.split())
        if 5 <= wc <= 12: score += 3
        elif 3 <= wc <= 15: score += 2
        else: score += 1
        if ql.startswith(('what','how')): score += 2
        elif ql.startswith(('who','when','where')): score += 1.5
        elif ql.startswith('why'): score += 2.5
        entity_names = [ent[0].lower() for ent in entities]
        for e in entity_names:
            if e in ql:
                score += 1
        if self.has_sufficient_content(question, content): score += 2
        for ind in ['implication','effect','impact','consequence','significance','analysis']:
            if ind in ql: score += 1.5
        return score
    def filter_and_rank_questions(self, questions, content, entities, max_questions=20):
        filtered = []
        for q in questions:
            if self.is_valid_question_format(q) and self.is_meaningful_question(q, entities) and self.has_sufficient_content(q, content):
                filtered.append(q)
        filtered = self.remove_duplicate_questions(filtered)
        scored = [(self.score_question_quality(q, content, entities), q) for q in filtered]
        scored.sort(reverse=True, key=lambda x: x[0])
        return [q for _, q in scored[:max_questions]]

class AdvancedQuestionGenerator:
    def __init__(self, t5_model, t5_tokenizer, qg_pipeline, device):
        self.t5_model = t5_model
        self.t5_tokenizer = t5_tokenizer
        self.qg_pipeline = qg_pipeline
        self.device = device
        self.filter = SmartQuestionFilter()
    def generate_contextual_wh_questions(self, entities, content):
        questions = []
        content_sents = nltk.sent_tokenize(content)
        templates = {
            "PERSON": [
                "What role did {} play in the events described?",
                "How is {} involved in this situation?",
                "What did {} accomplish according to the article?",
                "Why is {} significant in this context?"
            ],
            "ORG": [
                "What is the significance of {} in this article?",
                "How does {} contribute to the main topic?",
                "What actions did {} take according to the report?",
                "What is {}'s position on the matter discussed?"
            ],
            "GPE": [
                "What events occurred in {}?",
                "How does {} relate to the main story?",
                "What is the situation in {} according to the article?",
                "Why is {} mentioned in this context?"
            ],
            "DATE": [
                "What happened on {}?",
                "Why is {} significant in this timeline?",
                "What events are associated with {}?"
            ],
            "EVENT": [
                "What were the consequences of {}?",
                "How did {} affect the situation?",
                "What led to {}?"
            ]
        }
        for text, label in entities:
            if label in templates:
                ent_sents = [s for s in content_sents if text in s]
                if ent_sents:
                    for t in templates[label]:
                        questions.append(t.format(text))
        return questions
    def generate_inference_questions(self, content):
        questions = []
        causal_ind = ["because","due to","as a result","caused by","led to","resulted in"]
        if any(ind in content.lower() for ind in causal_ind):
            questions += [
                "What are the underlying causes mentioned in the article?",
                "What chain of events is described?",
                "How are different events connected in this story?"
            ]
        problem_ind = ["problem","issue","challenge","difficulty","crisis"]
        solution_ind = ["solution","resolve","address","tackle","fix"]
        has_problem = any(ind in content.lower() for ind in problem_ind)
        has_solution = any(ind in content.lower() for ind in solution_ind)
        if has_problem: questions.append("What problems or challenges are identified in the article?")
        if has_solution: questions.append("What solutions or approaches are suggested?")
        if has_problem and has_solution: questions.append("How effective might the proposed solutions be?")
        future_ind = ["will","expected","predict","forecast","likely","potential"]
        if any(ind in content.lower() for ind in future_ind):
            questions += [
                "What future developments are anticipated?",
                "What predictions are made in the article?",
                "What are the potential long-term effects?"
            ]
        return questions
    def generate_critical_thinking_questions(self, content):
        questions = [
            "What evidence is provided to support the main claims?",
            "What perspectives or viewpoints are presented?",
            "What information might be missing from this account?",
            "How reliable are the sources mentioned in the article?",
            "What assumptions underlie the arguments presented?",
            "How might different stakeholders view this situation?",
            "What are the broader implications of these events?",
            "What questions remain unanswered after reading this article?"
        ]
        if "research" in content.lower() or "study" in content.lower():
            questions += [
                "What methodology was used in the research described?",
                "What are the limitations of this study?",
                "How generalizable are these findings?"
            ]
        if any(w in content.lower() for w in ["policy","government","political"]):
            questions += [
                "What are the political implications of these events?",
                "How might this affect public policy?",
                "What are the different political perspectives on this issue?"
            ]
        return questions
    def generate_ml_questions_improved(self, content):
        questions = []
        sentences = nltk.sent_tokenize(content)
        important = sentences[:3] if len(sentences) >= 3 else sentences
        for sent in important:
            if len(sent.split()) > 10:
                prompts = [
                    f"generate question: {sent}",
                    f"create question about: {sent}",
                    f"what to ask about: {sent}"
                ]
                for prompt in prompts:
                    try:
                        input_ids = self.t5_tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True).to(self.device)
                        outputs = self.t5_model.generate(input_ids, max_length=50, num_beams=3, early_stopping=True, do_sample=False, pad_token_id=self.t5_tokenizer.eos_token_id)
                        q = self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
                        if q and len(q.split()) > 3:
                            if not q.endswith('?'):
                                q += '?'
                            questions.append(q)
                    except Exception:
                        continue
        return questions
    def generate_all_questions(self, entities, content, question_types=None, max_questions=20):
        if question_types is None:
            question_types = ["contextual_wh","inference","critical","ml"]
        all_q = []
        if "contextual_wh" in question_types:
            all_q += self.generate_contextual_wh_questions(entities, content)
        if "inference" in question_types:
            all_q += self.generate_inference_questions(content)
        if "critical" in question_types:
            all_q += self.generate_critical_thinking_questions(content)
        if "ml" in question_types:
            all_q += self.generate_ml_questions_improved(content)
        return self.filter.filter_and_rank_questions(all_q, content, entities, max_questions)

def save_questions_to_file(questions, title, format_type="txt"):
    if format_type == "txt":
        content = f"Questions for: {title}\n" + "="*50 + "\n\n"
        content += "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
        return content
    elif format_type == "json":
        data = {"title": title, "questions": questions, "generated_at": str(pd.Timestamp.now())}
        return json.dumps(data, indent=2)

# -----------------------
# Offline TTS helper (WAV)
# -----------------------
def tts_offline_to_wav_bytes(text: str):
    """
    Generate speech for the given text using pyttsx3 into a WAV file and return bytes.
    Streamlit will rerun on button clicks; use session_state to persist questions.
    """
    if not PYTTSX3_AVAILABLE:
        st.error("pyttsx3 not installed. Run: pip install pyttsx3")
        return None
    text = (text or "").strip()
    if not text:
        st.warning("Nothing to speak.")
        return None
    try:
        # Use WAV for maximum compatibility with pyttsx3 engines
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            wav_path = tmp.name
        engine = pyttsx3.init()
        # Optional: tune voice rate/volume as needed
        # engine.setProperty('rate', 180)
        # engine.setProperty('volume', 1.0)
        engine.save_to_file(text, wav_path)
        engine.runAndWait()
        with open(wav_path, "rb") as f:
            audio_bytes = f.read()
        os.unlink(wav_path)
        return audio_bytes
    except Exception as e:
        st.error(f"TTS generation failed: {e}")
        return None

# -----------------------
# UI
# -----------------------
st.set_page_config(page_title="Advanced Question Generator", layout="wide")
st.title("🤖 Advanced Automatic Question Generator")
st.markdown("Generate high-quality, contextual questions from news articles using advanced NLP techniques.")

# Load models with error handling
try:
    t5_model, t5_tokenizer, qg_pipeline, device = load_models()
    question_generator = AdvancedQuestionGenerator(t5_model, t5_tokenizer, qg_pipeline, device)
except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    st.stop()

# Sidebar configuration
st.sidebar.title("⚙️ Configuration")
question_types = st.sidebar.multiselect(
    "Select Question Types:",
    ["contextual_wh", "inference", "critical", "ml"],
    default=["contextual_wh", "inference", "critical"],
    help="Choose which types of questions to generate"
)
max_questions = st.sidebar.slider("Maximum Questions:", min_value=5, max_value=30, value=15)
show_scores = st.sidebar.checkbox("Show Question Quality Scores", value=False)

# Article input
st.markdown("### 📝 Article Input")
url = st.text_input("🔗 Enter News Article URL", placeholder="https://www.example.com/article")

col1, col2 = st.columns([2, 1])

with col1:
    if st.button("🚀 Generate Questions", type="primary"):
        if url.strip():
            with st.spinner("Processing article and generating questions..."):
                try:
                    article_title, article_content = fetch_article_text(url)
                    if not article_title or not article_content:
                        st.error("Could not extract article content. Please check the URL.")
                        st.stop()
                    entities = extract_named_entities(article_content)
                    questions = question_generator.generate_all_questions(
                        entities, article_content, question_types, max_questions
                    )
                    if not questions:
                        st.warning("No valid questions could be generated from this article.")
                        st.stop()

                    # Persist into session so buttons don't 'clear' results on rerun
                    st.session_state.questions = questions
                    st.session_state.article_title = article_title

                    st.success(f"Generated {len(questions)} high-quality questions!")

                    with st.expander("📰 Article Information", expanded=False):
                        st.markdown(f"**Title:** {article_title}")
                        st.markdown(f"**Content Length:** {len(article_content)} characters")
                        st.markdown(f"**Named Entities Found:** {len(entities)}")
                        if entities:
                            entity_display = ", ".join([f"{ent[0]} ({ent[1]})" for ent in entities[:10]])
                            if len(entities) > 10:
                                entity_display += f"... and {len(entities)-10} more"
                            st.markdown(f"**Key Entities:** {entity_display}")

                    # Display questions + capture exactly what's shown for TTS
                    st.markdown("### 🎯 Generated Questions")
                    displayed_lines = []
                    if show_scores:
                        filter_obj = SmartQuestionFilter()
                        for i, q in enumerate(questions, 1):
                            score = filter_obj.score_question_quality(q, article_content, entities)
                            line = f"{i}. {q} (Quality Score: {score:.1f})"
                            st.markdown(f"**{line}**")
                            displayed_lines.append(f"Question {i}: {q}. Quality score {score:.1f}.")
                    else:
                        for i, q in enumerate(questions, 1):
                            line = f"{i}. {q}"
                            st.markdown(line)
                            displayed_lines.append(f"Question {i}: {q}")

                    # Save the displayed text in session_state for TTS use on rerun
                    st.session_state.displayed_text = "\n".join(displayed_lines)

                    # Export options
                    st.markdown("### 📥 Export Options")
                    col_exp1, col_exp2 = st.columns(2)
                    with col_exp1:
                        txt_data = save_questions_to_file(questions, article_title, "txt")
                        st.download_button(
                            label="📄 Download as Text",
                            data=txt_data,
                            file_name=f"questions_{article_title[:30].replace(' ', '_')}.txt",
                            mime="text/plain"
                        )
                    with col_exp2:
                        json_data = save_questions_to_file(questions, article_title, "json")
                        st.download_button(
                            label="📊 Download as JSON",
                            data=json_data,
                            file_name=f"questions_{article_title[:30].replace(' ', '_')}.json",
                            mime="application/json"
                        )
                except Exception as e:
                    st.error(f"Error processing article: {str(e)}")
        else:
            st.warning("Please enter a valid article URL.")

    # -----------------------
    # TTS controls (work with session_state to avoid 'refresh loss')
    # -----------------------
    if st.session_state.questions:
        st.markdown("### 🔊 Text-to-Speech")

        # 1) Speak exactly what is shown above (including scores if shown)
        if st.button("▶️ Speak What's Shown"):
            audio_bytes = tts_offline_to_wav_bytes(st.session_state.displayed_text)
            if audio_bytes:
                st.audio(audio_bytes, format="audio/wav")

        # 2) Speak all questions (clean format, with separators)
        if st.button("▶️ Speak All Questions (Clean)"):
            # Build a clean narration: "Question 1: ... Next question. ..."
            parts = []
            for i, q in enumerate(st.session_state.questions, 1):
                parts.append(f"Question {i}: {q}")
                if i < len(st.session_state.questions):
                    parts.append("Next question.")
            clean_text = " ".join(parts)
            audio_bytes = tts_offline_to_wav_bytes(clean_text)
            if audio_bytes:
                st.audio(audio_bytes, format="audio/wav")

        # 3) Speak a selected question
        idx = st.number_input(
            "Speak a specific question number:",
            min_value=1,
            max_value=len(st.session_state.questions),
            value=1,
            step=1
        )
        if st.button("▶️ Speak Selected Question"):
            q_text = f"Question {idx}: {st.session_state.questions[idx-1]}"
            audio_bytes = tts_offline_to_wav_bytes(q_text)
            if audio_bytes:
                st.audio(audio_bytes, format="audio/wav")

with col2:
    st.markdown("### 💡 Tips")
    st.info("""
    - Click 'Generate Questions' to fetch the article and produce questions.
    - After generation, use the Text-to-Speech buttons to hear:
      • Exactly what's displayed (including scores if enabled)  
      • All questions in a clean narration  
      • A single selected question
    - Streamlit reruns the script on every button click; we keep results using session_state.
    """)
    if not PYTTSX3_AVAILABLE:
        st.warning("Text-to-Speech requires pyttsx3. Install it with: pip install pyttsx3\nOn Linux you may also need: sudo apt-get install espeak")

# Footer
st.markdown("---")
st.caption("TTS uses offline pyttsx3 to avoid network/SSL issues. If you need different voices or languages, configure your OS speech settings or pyttsx3 voices.")