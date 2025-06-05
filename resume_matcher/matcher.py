import spacy
import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def clean_text(text):
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc if token.is_alpha and not token.is_stop])

def get_similarity(job_desc, resume_text):
    cleaned_job = clean_text(job_desc)
    cleaned_resume = clean_text(resume_text)
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([cleaned_job, cleaned_resume])
    score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return round(score * 100, 2)  # Return as percentage
