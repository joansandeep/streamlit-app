import os
import re
import difflib
import logging
import time
import random
import requests
import pandas as pd
import streamlit as st
from groq import Groq
from src.data_processor import DataProcessor
from src.embedder import Embedder


# Setup a logger
logger = logging.getLogger("rag_pipeline_logger")
handler = logging.FileHandler("rag_pipeline_queries.log", mode="a", encoding="utf-8")
formatter = logging.Formatter('%(asctime)s - QUERY: %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class RAGPipeline:
    def __init__(self, faiss_path, data_path):
        # Allow override via env, fallback to default
        self.model_name = st.secrets.get("GROQ_MODEL_NAME", "llama-3.1-8b-instant")
        self.faiss_path = faiss_path
        self.data_path = data_path


        # Load and preprocess data
        processor = DataProcessor(data_path=self.data_path, save_path=self.data_path)
        self.df = processor.load_data()
        processor.preprocess()
        self.df = processor.df


        # Medicine vocab
        self.generic_meds = self.df['Generic Name'].dropna().tolist()
        self.brand_names = self.df['Brand Name'].dropna().tolist()


        # Vector store (FAISS)
        self.embedder = Embedder()
        self.embedder.load_vector_store(faiss_load_path=self.faiss_path, data_df=self.df)


        # Initialize API clients
        self._setup_api_clients()


        # Precompile regex for medicine lookup
        self._med_patterns = {
            'generic': [re.compile(rf"\b{re.escape(m.lower())}\b") for m in self.generic_meds],
            'brand': [re.compile(rf"\b{re.escape(m.lower())}\b") for m in self.brand_names],
        }


        # Add known problematic brands for edge cases
        self.known_problematic_brands = {
            "ximafen 600mg capsule": "Ximafen 600mg Capsule",
            "danaibu 400mg tablet": "Danaibu 400mg Tablet", 
            "fenceta 400mg/325mg tablet": "Fenceta 400mg/325mg Tablet",
            "novolid 100mg tablet": "Novolid 100mg Tablet",
            "crocin advance 500mg tablet": "Crocin Advance 500mg Tablet"
        }


    def _setup_api_clients(self):
        """Setup primary and fallback API clients"""
        # Primary: Groq client
        self.groq_api_key = st.secrets["GROQ_API_KEY"]
        if self.groq_api_key:
            self.groq_client = Groq(api_key=self.groq_api_key)
        else:
            self.groq_client = None
            logger.warning("GROQ_API_KEY not found")


        # Fallback: OpenRouter client
        self.openrouter_api_key = st.secrets["OPENROUTER_API_KEY"]
        if not self.openrouter_api_key:
            logger.warning("OPENROUTER_API_KEY not found - no fallback available")


    def _call_groq_api(self, prompt, temperature=0.7, max_tokens=500, top_p=0.9):
        """Call Groq API with the original method"""
        if not self.groq_client:
            raise Exception("Groq client not initialized")
            
        result = self.groq_client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p
        )
        choice = result.choices[0].message
        if isinstance(choice, dict):
            return choice.get("content", "").strip()
        return choice.content.strip()


    def _call_openrouter_api(self, prompt, temperature=0.7, max_tokens=500, top_p=0.9):
        """Fallback to OpenRouter API with working Llama 3.1 8B model"""
        if not self.openrouter_api_key:
            raise Exception("OpenRouter API key not available")


        openrouter_model = "meta-llama/llama-3.1-8b-instruct"
        
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
            "X-Title": "RAG Medicine Assistant"
        }
        
        data = {
            "model": openrouter_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p
        }
        
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
        else:
            raise Exception(f"OpenRouter API error: {response.status_code} - {response.text}")


    def _call_model(self, prompt, temperature=0.7, max_tokens=500, top_p=0.9, max_retries=3):
        """Enhanced model calling with automatic fallback"""
        last_error = None
        
        # First, try Groq API with retries
        if self.groq_client:
            for attempt in range(max_retries):
                try:
                    if attempt > 0:
                        time.sleep(random.uniform(0.2, 0.5))
                    
                    return self._call_groq_api(prompt, temperature, max_tokens, top_p)
                    
                except Exception as e:
                    last_error = e
                    error_str = str(e)
                    
                    if "429" in error_str and attempt < max_retries - 1:
                        wait_time = (2 ** attempt) + random.uniform(0.5, 1.5)
                        logger.warning(f"Groq rate limited, waiting {wait_time:.1f}s...")
                        time.sleep(wait_time)
                        continue
                    
                    logger.warning(f"Groq API failed (attempt {attempt + 1}): {error_str}")
                    break
        
        # Fallback to OpenRouter API
        if self.openrouter_api_key:
            try:
                logger.info("Falling back to OpenRouter API (Llama 3.1 8B)...")
                return self._call_openrouter_api(prompt, temperature, max_tokens, top_p)
            except Exception as e:
                last_error = e
                logger.error(f"OpenRouter API also failed: {str(e)}")
        
        return f"Sorry, both primary and fallback APIs are unavailable: {str(last_error)}"


    # ========== PDF PROCESSING METHODS ===========
    
    def extract_text_from_pdf(self, uploaded_file):
        """Extract text from uploaded PDF file"""
        try:
            import PyPDF2
            from io import BytesIO
            
            pdf_bytes = uploaded_file.read()
            pdf_stream = BytesIO(pdf_bytes)
            
            pdf_reader = PyPDF2.PdfReader(pdf_stream)
            text = ""
            
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
            
            return text.strip() if text.strip() else "No text could be extracted from this PDF."
            
        except ImportError:
            return "Error: PyPDF2 library not installed. Please install it with: pip install PyPDF2"
        except Exception as e:
            return f"Error extracting text from PDF: {str(e)}"


    def process_pdf_query(self, pdf_text, user_query):
        """Process queries about PDF content with medicine database integration"""
        if not pdf_text or pdf_text.startswith("Error"):
            return "Sorry, I couldn't process the PDF content."
        
        found_medicines = self._find_medicines_in_text(pdf_text)
        
        if found_medicines:
            medicine_context = self._get_medicine_details(found_medicines)
            
            prompt = f"""
You are a medical AI assistant. Based on the PDF content and medicine database information provided below, answer the user's question comprehensively.


PDF Content (first 1500 characters):
{pdf_text[:1500]}...


Medicine Information from Database:
{medicine_context}


User Question: {user_query}


Please provide a detailed answer using both the PDF content and the medicine database. If any medicines are mentioned:
1. Explain what they are used for
2. Mention their active ingredients (salts)
3. Note any important side effects or precautions
4. Always recommend consulting a healthcare professional for medical advice


Answer:"""
            
            return self._call_model(prompt, max_tokens=800)
        
        else:
            prompt = f"""
Based on the following PDF content, please answer the user's question:


PDF Content:
{pdf_text[:2000]}


User Question:
{user_query}


Provide a helpful and informative answer based on the content. If this appears to be medical content, remind the user to consult healthcare professionals for medical advice.


Answer:"""
            
            return self._call_model(prompt, max_tokens=600)


    def _find_medicines_in_text(self, text):
        """Find medicines mentioned in text using the medicine database"""
        found_medicines = []
        text_lower = text.lower()
        
        for brand in self.brand_names:
            if brand and len(brand) > 2:
                brand_lower = brand.lower()
                if brand_lower in text_lower:
                    found_medicines.append(('brand', brand))
        
        for generic in self.generic_meds:
            if generic and len(generic) > 2:
                generic_lower = generic.lower()
                if generic_lower in text_lower:
                    found_medicines.append(('generic', generic))
        
        seen = set()
        unique_medicines = []
        for med_type, med_name in found_medicines:
            if med_name not in seen:
                seen.add(med_name)
                unique_medicines.append((med_type, med_name))
        
        return unique_medicines[:10]


    def _get_medicine_details(self, found_medicines):
        """Get detailed information about found medicines from database"""
        medicine_context = ""
        
        for med_type, med_name in found_medicines:
            if med_type == 'brand':
                med_info = self.df[self.df['Brand Name'] == med_name]
            else:
                med_info = self.df[self.df['Generic Name'] == med_name]
            
            if not med_info.empty:
                med_data = med_info.iloc[0]
                medicine_context += f"""
Medicine: {med_data.get('Brand Name', med_data.get('Generic Name', 'Unknown'))}
Type: {med_type.title()}
Active Salt: {med_data.get('Salt', 'Not specified')}
Uses/Indications: {med_data.get('Uses', med_data.get('Indication', 'Not specified'))}
Side Effects: {med_data.get('Side_Effects', med_data.get('Side Effects', 'Not specified'))}
Manufacturer: {med_data.get('Manufacturer', 'Not specified')}
Strength: {med_data.get('Strength', 'Not specified')}
---
"""
        
        return medicine_context if medicine_context else "No detailed medicine information found in database."


    def _extract_medicine_types(self, query):
        """Enhanced medicine extraction with edge case handling"""
        q = query.lower()
        matched_brand = []
        matched_generic = []
        
        for query_form, actual_brand in self.known_problematic_brands.items():
            if query_form in q and actual_brand in self.df['Brand Name'].values:
                matched_brand.append(actual_brand)
                break
        
        if not matched_brand:
            matched_brand = [med for med, pat in zip(self.brand_names, self._med_patterns['brand']) if pat.search(q)]
        
        matched_generic = [med for med, pat in zip(self.generic_meds, self._med_patterns['generic']) if pat.search(q)]
        
        return matched_generic, matched_brand


    def _extract_brand_name_from_query(self, query):
        """Extract brand name from queries for alternative detection"""
        query_lower = query.lower()
        
        if "alternative to" in query_lower:
            brand_part = query_lower.split("alternative to")[-1].strip(" ?.,")
            
            if brand_part in self.known_problematic_brands:
                return self.known_problematic_brands[brand_part]
            
            for brand in self.brand_names:
                if brand.lower() == brand_part or brand_part in brand.lower():
                    return brand
        
        return None


    def _extract_brand_name_from_query_enhanced(self, query):
        """Enhanced brand name extraction from queries"""
        query_lower = query.lower()
        
        if "alternative to" in query_lower:
            brand_part = query_lower.split("alternative to")[-1].strip(" ?.,")
            
            if brand_part in self.known_problematic_brands:
                return self.known_problematic_brands[brand_part]
            
            for brand in self.brand_names:
                if brand.lower() == brand_part:
                    return brand
            
            for brand in self.brand_names:
                if brand_part in brand.lower() or brand.lower() in brand_part:
                    return brand
        
        return None


    def _summarize_pdf(self, context, user_query):
        prompt = (
            "You are a helpful assistant. Use ONLY the provided report text. "
            "Give a concise summary or answer.\n\n"
            f"Context:\n{context}\n\nUser Question:\n{user_query}\n\nAnswer:"
        )
        return self._call_model(prompt)


    def _answer_with_context_only(self, context, user_query):
        prompt = (
            "Use ONLY the following context to answer. If the answer is not present, say so.\n\n"
            f"Context:\n{context}\n\nUser Question:\n{user_query}\n\nAnswer:"
        )
        return self._call_model(prompt)


    def _compose_medicine_summary_from_pdf(self, found_meds, context):
        df_filtered = self.df[
            (self.df['Generic Name'].isin(found_meds)) |
            (self.df['Brand Name'].isin(found_meds))
        ]
        if df_filtered.empty:
            return f"The following medicines were found in the PDF: {', '.join(found_meds)}. Please consult a healthcare professional for more details."
        med_info = df_filtered.to_dict(orient='records')
        lines = []
        for info in med_info:
            line = (
                f"{info.get('Brand Name') or info.get('Generic Name','')} "
                f"({info.get('Salt','')}) - Contains {info.get('Generic Name','')} "
                f"({info.get('Strength','')}) which is used to {info.get('Indication','').lower()}."
            )
            lines.append(line)
        advice = ("However, please note that for any medical condition, always consult a qualified "
                  "healthcare professional for diagnosis and treatment.")
        return f"Based on the PDF, the following medicines are prescribed: {' '.join(lines)} {advice}"


    def _get_enhanced_context_with_exact_data(self, query, matched_brands):
        """Enhanced context injection for better Llama 3 responses"""
        retrieved_chunks = self.embedder.retrieve(query, top_k=8)
        
        if matched_brands and ('what is' in query.lower() or 'contains' in query.lower()):
            brand = matched_brands[0]
            brand_info = self.df[self.df['Brand Name'] == brand]
            if not brand_info.empty:
                info = brand_info.iloc[0]
                
                exact_info = (
                    f"MEDICINE FACT: {brand} is a pharmaceutical brand that contains "
                    f"the active ingredient {info['Salt']}. It is manufactured by "
                    f"{info.get('Manufacturer', 'a pharmaceutical company')} and is "
                    f"used for {info.get('Indication', 'medical treatment').lower()}."
                )
                
                retrieved_chunks.insert(0, exact_info)
        
        return retrieved_chunks


    def _is_multi_query(self, query):
        """Detect if query contains multiple questions"""
        multi_indicators = [
            ' and what ', ' and its ', ' what are ', 
            '? and', '? what', '. what', '; what'
        ]
        return any(indicator in query.lower() for indicator in multi_indicators)


    def _handle_multi_query(self, query, context=None):
        """Handle queries with multiple sub-questions"""
        sub_queries = re.split(r'\?|\band\s+(?=what|how|where|when|why|which)', query, flags=re.IGNORECASE)
        sub_queries = [q.strip().rstrip('.,;') for q in sub_queries if q.strip()]
        
        if len(sub_queries) <= 1:
            return self._run_single_query(query, context)
        
        answers = []
        for i, sq in enumerate(sub_queries):
            if not sq.endswith('?'):
                sq += '?'
            
            answer = self._run_single_query(sq, context)
            answers.append(f"**Query {i+1}:** {sq}\n**Answer:** {answer}")
        
        return "\n\n".join(answers)


    def run(self, user_query: str, context: str = None) -> str:
        if not user_query or not user_query.strip():
            return "Please provide a question."


        user_query_clean = user_query.strip()


        # Check for multi-part queries
        if self._is_multi_query(user_query_clean):
            return self._handle_multi_query(user_query_clean, context)
        else:
            return self._run_single_query(user_query_clean, context)


    def _run_single_query(self, user_query: str, context: str = None) -> str:
        """Process a single query"""
        user_query_clean = user_query.strip()
        user_query_lower = user_query_clean.lower()


        # Greetings
        if user_query_lower in {"hi", "hello", "hey", "good morning", "good evening"}:
            return "Hello! How can I help you with medicine information today?"


        # If PDF context is provided
        if context:
            summary_keywords = {"summarize", "summary", "report", "overview"}
            if any(kw in user_query_lower for kw in summary_keywords):
                return self._summarize_pdf(context, user_query_clean)


            medicine_request_keywords = {"medicine", "medicines", "drugs", "prescribed"}
            if any(kw in user_query_lower for kw in medicine_request_keywords) and (
                "used" in user_query_lower or "in this pdf" in user_query_lower or "in this report" in user_query_lower
            ):
                found_meds = []
                ctx_lower = context.lower()
                for med in set(self.generic_meds + self.brand_names):
                    if med and med.lower() in ctx_lower:
                        found_meds.append(med)
                if found_meds:
                    return self._compose_medicine_summary_from_pdf(found_meds, context)
                return "No medicines were found in the PDF."


            matched_generic, matched_brand = self._extract_medicine_types(user_query_clean)
            found_specific = []
            ctx_lower = context.lower()
            for med in matched_generic + matched_brand:
                if med.lower() in ctx_lower:
                    found_specific.append(med)
            if found_specific:
                df_filtered = self.df[
                    (self.df['Generic Name'].isin(found_specific)) |
                    (self.df['Brand Name'].isin(found_specific))
                ]
                if not df_filtered.empty:
                    med_info = df_filtered.to_dict(orient='records')
                    details = '\n'.join([str(info) for info in med_info])
                    prompt = (
                        "The following medicine data was found in the report/database:\n"
                        f"{details}\n\nUser Question:\n{user_query_clean}\n\nAnswer succinctly:"
                    )
                    return self._call_model(prompt)


            return self._answer_with_context_only(context, user_query_clean)


        # Disclaimers for symptom/advice queries
        symptom_keywords = [
            "i have", "my symptoms", "i feel", "i am suffering",
            "pain", "headache", "fever", "vomiting", "cough", "nausea"
        ]
        medical_advice_keywords = ['diagnose', 'symptoms', 'what should i take', 'sick']
        disclaimer = ""
        if any(sk in user_query_lower for sk in symptom_keywords) or any(mk in user_query_lower for mk in medical_advice_keywords):
            disclaimer = (
                "Note: I am an AI assistant, not a healthcare professional. "
                "Always consult a qualified provider for diagnosis and treatment.\n\n"
            )


        # Combined price + alternatives logic
        if 'alternative' in user_query_lower and any(pk in user_query_lower for pk in ['price','cost','cheapest']):
            matched_generic, matched_brand = self._extract_medicine_types(user_query_clean)
            if matched_brand:
                brand = matched_brand[0]
                dfb = self.df[self.df['Brand Name'] == brand]
                price_val = dfb['Price'].iloc[0] if not dfb.empty and 'Price' in dfb else "unknown"


                alt_df = self.get_alternatives(brand)
                if not alt_df.empty:
                    cheapest_brand, min_price = self.get_cheapest_alternative(brand)
                    alt_list = [
                        f"{row['Brand Name']} (₹{row['Price']})"
                        for _, row in alt_df.head(5).iterrows()
                    ]
                    alt_text = ", ".join(alt_list)
                    return (
                        f"{brand} costs ₹{price_val}.\n"
                        f"Alternatives with same salt composition: {alt_text}.\n"
                        f"The cheapest alternative is {cheapest_brand} at ₹{min_price:.2f}."
                    )
                return f"{brand} costs ₹{price_val}. No alternatives found."


        # Price-only lookup
        price_keywords = ['price', 'cheapest', 'lowest price', 'least cost', 'cost effective']
        if any(pk in user_query_lower for pk in price_keywords):
            matched_generic, matched_brand = self._extract_medicine_types(user_query_clean)
            if matched_generic or matched_brand:
                df_filtered = self.df[
                    (self.df['Generic Name'].isin(matched_generic)) |
                    (self.df['Brand Name'].isin(matched_brand))
                ]
            else:
                df_filtered = self.df
            if df_filtered.empty:
                return "Sorry, no matching price information found."
            df_temp = df_filtered.copy()
            try:
                df_temp['Price'] = (
                    df_temp['Price']
                    .astype(str)
                    .str.replace('₹', '', regex=False)
                    .str.replace(',', '', regex=False)
                    .astype(float)
                )
            except Exception:
                return "Price data is unavailable or malformed."
            min_price = df_temp['Price'].min()
            meds_rows = df_temp[df_temp['Price'] == min_price]
            names = set(
                meds_rows['Generic Name'].dropna().tolist() +
                meds_rows['Brand Name'].dropna().tolist()
            )
            meds_list = ', '.join(sorted(names))
            return f"The cheapest option(s): {meds_list} at ₹{min_price:g}"


        matched_generic, matched_brand = self._extract_medicine_types(user_query_clean)
        intro_notes = ""
        if matched_generic or matched_brand:
            parts = []
            if matched_generic:
                parts.append(f"generic: {', '.join(matched_generic)}")
            if matched_brand:
                parts.append(f"brand: {', '.join(matched_brand)}")
            salts_info = []
            for b in matched_brand:
                salts = self.df[self.df['Brand Name'] == b]['Salt'].dropna().unique()
                if salts.size > 0:
                    salts_info.append(f"Brand '{b}' contains salt(s): {', '.join(salts)}")
            intro_notes = f"Note: Query mentions {', '.join(parts)}.\n"
            if salts_info:
                intro_notes += "\n".join(salts_info) + "\n"


        # Enhanced alternative detection
        if 'alternative' in user_query_lower:
            matched_generic, matched_brand = self._extract_medicine_types(user_query_clean)
            brand = matched_brand[0] if matched_brand else self._extract_brand_name_from_query_enhanced(user_query_clean)
            if brand:
                if brand not in self.df['Brand Name'].values:
                    fuzzy_match = self._find_brand_fuzzy_match(brand)
                    if not fuzzy_match.empty:
                        brand = fuzzy_match.iloc[0]['Brand Name']
                    else:
                        return f"Sorry, I couldn't find information about '{brand}' in the database."


                strength_keys = ['strength', 'dosage', 'dose', 'mg']
                asking_strength = any(k in user_query_lower for k in strength_keys)
                if any(pk in user_query_lower for pk in ['cheap', 'cheaper', 'lowest price']):
                    return self.answer_alternatives_and_cheapest(brand)
                if not asking_strength:
                    alt_brands_with_prices = self.get_alternatives(brand)
                    if not alt_brands_with_prices.empty:
                        alt_list = []
                        for _, row in alt_brands_with_prices.iterrows():
                            price_clean = str(row['Price']).replace('₹','').replace(',','')
                            try:
                                price_num = float(price_clean)
                                alt_list.append(f"{row['Brand Name']} (₹{price_num:.2f})")
                            except:
                                alt_list.append(f"{row['Brand Name']} ({row['Price']})")
                        cheapest_brand, min_price = self.get_cheapest_alternative(brand)
                        return (
                            f"Alternatives to {brand}: {', '.join(alt_list)}.\n"
                            f"The cheapest is {cheapest_brand} at ₹{min_price:.2f}."
                        )
                    return f"No alternatives found for {brand}."
            return "Please specify a valid medicine name."


        # Enhanced RAG retrieval
        retrieved_chunks = self._get_enhanced_context_with_exact_data(user_query_clean, matched_brand if 'matched_brand' in locals() else [])
        context_text = "\n\n".join(retrieved_chunks[:8]) if retrieved_chunks else "No relevant data found."


        prompt = (
            f"{disclaimer}{intro_notes}"
            "You are a helpful medical assistant. Use ONLY the provided context.\n\n"
            f"Context:\n{context_text}\n\n"
            f"User Question:\n{user_query_clean}\n\n"
            "Answer:"
        )
        return self._call_model(prompt)


    # UTILITY METHODS


    def normalize_text(self, text):
        if not text:
            return ""
        text = text.lower()
        text = re.sub(r'\([^)]*\)', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = ' '.join(text.split())
        return text


    def _fuzzy_match(self, expected, text, threshold=0.5):
        expected_norm = self.normalize_text(expected)
        text_norm = self.normalize_text(text)
        if not expected_norm or not text_norm:
            return False
        seq = difflib.SequenceMatcher(None, expected_norm, text_norm)
        similarity = seq.ratio()
        return similarity >= threshold


    def _extract_salts_from_answer(self, answer):
        pattern = r'([A-Za-z ]+)(?:\s*\(\d+mg\))?'
        salts_found = re.findall(pattern, answer)
        return list(set([s.lower().strip() for s in salts_found]))


    def log_query_and_response(self, query, response):
        logger.info(f"{query} || RESPONSE: {response}")


    def run_with_logging(self, user_query: str, context: str = None) -> str:
        response = self.run(user_query, context)
        self.log_query_and_response(user_query, response)
        return response


    def evaluate_answer_correctness(self, expected_salt, answer, alt_brands=None, threshold=0.4):
        answer_lower = answer.lower()
        expected_lower = expected_salt.lower()
        
        if "error occurred" in answer_lower or "rate limit" in answer_lower or "unavailable" in answer_lower:
            return False
        
        no_info_phrases = [
            "no information", "no alternatives", "not enough information",
            "i do not have", "unknown", "not available", "not provided",
            "consult a doctor", "consult a pharmacist"
        ]
        if any(phrase in answer_lower for phrase in no_info_phrases) and (not alt_brands or alt_brands == ["No alternatives found"]):
            return True

        medicine_name = re.sub(r'\s*\([^)]*\)', '', expected_salt).strip().lower()
        
        patterns_to_check = [
            expected_lower,
            medicine_name,
            expected_salt.replace('(', '').replace(')', '').lower(),
        ]
        
        for pattern in patterns_to_check:
            if pattern in answer_lower:
                return True
        
        if medicine_name in answer_lower:
            dosage_match = re.search(r'\((\d+mg)\)', expected_salt)
            if dosage_match:
                dosage = dosage_match.group(1).lower()
                if dosage in answer_lower or dosage.replace('mg', '') in answer_lower:
                    return True


        expected_norm = self.normalize_text(expected_salt)
        answer_norm = self.normalize_text(answer)
        if expected_norm in answer_norm:
            return True


        if alt_brands:
            alt_brands_lower = [b.lower() for b in alt_brands]
            if any(alt in answer_lower for alt in alt_brands_lower):
                return True


        extracted_salts = self._extract_salts_from_answer(answer)
        for salt in extracted_salts:
            if self._fuzzy_match(expected_salt.lower(), salt, threshold=threshold):
                return True


        if '+' in expected_salt:
            components = [comp.strip().split('(')[0].strip().lower() for comp in expected_salt.split('+')]
            if all(comp in answer_lower for comp in components):
                return True


        return False


    def get_alternatives(self, brand_name):
        """Enhanced method to find alternatives with same salt composition"""
        
        brand_rows = self.df[self.df['Brand Name'] == brand_name]
        if brand_rows.empty:
            brand_rows = self._find_brand_fuzzy_match(brand_name)
            if brand_rows.empty:
                return pd.DataFrame()
        
        original_salt = brand_rows.iloc[0]['Salt']
        
        alternatives_df = self.df[
            (self.df['Salt'] == original_salt) & 
            (self.df['Brand Name'] != brand_name)
        ].copy()
        
        if alternatives_df.empty:
            return pd.DataFrame()
        
        alternatives_df = self._clean_price_data(alternatives_df)
        alternatives_df = alternatives_df.sort_values('clean_price')
        
        return alternatives_df[['Brand Name', 'Price']].reset_index(drop=True)


    def _find_brand_fuzzy_match(self, brand_name):
        """Fuzzy matching for brand names that might have slight variations"""
        brand_lower = brand_name.lower()
        
        for _, row in self.df.iterrows():
            if pd.isna(row['Brand Name']):
                continue
            
            db_brand_lower = row['Brand Name'].lower()
            
            if brand_lower == db_brand_lower:
                return self.df[self.df['Brand Name'] == row['Brand Name']]
            
            brand_base = re.sub(r'\d+mg[/\d+mg]*', '', brand_lower).strip()
            db_brand_base = re.sub(r'\d+mg[/\d+mg]*', '', db_brand_lower).strip()
            
            if brand_base and db_brand_base and brand_base == db_brand_base:
                return self.df[self.df['Brand Name'] == row['Brand Name']]
        
        return pd.DataFrame()


    def _clean_price_data(self, df):
        """Clean and standardize price data for sorting"""
        df = df.copy()
        
        df['clean_price'] = (
            df['Price']
            .astype(str)
            .str.replace('₹', '', regex=False)
            .str.replace(',', '', regex=False)
            .str.replace('Rs.', '', regex=False)
            .str.strip()
        )
        
        df['clean_price'] = pd.to_numeric(df['clean_price'], errors='coerce')
        df = df.dropna(subset=['clean_price'])
        
        return df


    def get_cheapest_alternative(self, brand_name):
        alt_brands = self.get_alternatives(brand_name)
        if len(alt_brands) == 0:
            return None, None
        alt_brands = alt_brands.copy()
        alt_brands['clean_price'] = alt_brands['Price'].astype(str).str.replace('₹', '', regex=False).str.replace(',', '', regex=False).astype(float)
        min_price = alt_brands['clean_price'].min()
        cheapest = alt_brands[alt_brands['clean_price'] == min_price]
        return cheapest['Brand Name'].iloc[0], min_price


    def answer_alternatives_and_cheapest(self, brand_name):
        alt_brands = self.get_alternatives(brand_name)
        if len(alt_brands) == 0:
            return f"No alternatives found for {brand_name}."
        alt_list = alt_brands['Brand Name'].unique().tolist()
        cheapest_brand, min_price = self.get_cheapest_alternative(brand_name)
        alt_text = ", ".join(alt_list)
        response = (
            f"Alternatives to {brand_name} are: {alt_text}.\n"
            f"The cheapest alternative is {cheapest_brand} at ₹{min_price:.2f}."
        )
        return response

    def _is_multi_query(self, query):
        """Detect if query contains multiple questions"""
        multi_indicators = [
            ' and what ', ' and its ', ' what are ', 
            '? and', '? what', '. what', '; what'
        ]
        return any(indicator in query.lower() for indicator in multi_indicators)


    def _handle_multi_query(self, query, context=None):
        """Handle queries with multiple sub-questions"""
        sub_queries = re.split(r'\?|\band\s+(?=what|how|where|when|why|which)', query, flags=re.IGNORECASE)
        sub_queries = [q.strip().rstrip('.,;') for q in sub_queries if q.strip()]
        
        if len(sub_queries) <= 1:
            return self._run_single_query(query, context)
        
        answers = []
        for i, sq in enumerate(sub_queries):
            if not sq.endswith('?'):
                sq += '?'
            
            answer = self._run_single_query(sq, context)
            answers.append(f"**Query {i+1}:** {sq}\n**Answer:** {answer}")
        
        return "\n\n".join(answers)


    def run(self, user_query: str, context: str = None) -> str:
        if not user_query or not user_query.strip():
            return "Please provide a question."


        user_query_clean = user_query.strip()


        # Check for multi-part queries
        if self._is_multi_query(user_query_clean):
            return self._handle_multi_query(user_query_clean, context)
        else:
            return self._run_single_query(user_query_clean, context)
