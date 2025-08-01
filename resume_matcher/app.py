from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer, util
import fitz  # PyMuPDF

app = Flask(__name__)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Define a basic skill set
COMMON_SKILLS = [
    "python", "java", "sql", "mongodb", "machine learning", "deep learning",
    "excel", "power bi", "flask", "django", "nlp", "react", "git", "docker",
    "aws", "tensorflow", "pandas", "numpy", "linux", "html", "css", "javascript"
]

def extract_text(pdf_file):
    text = ""
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text.strip().lower()

def extract_skills(text):
    return [skill for skill in COMMON_SKILLS if skill in text]

@app.route("/", methods=["GET", "POST"])
def index():
    results = {}
    error = None

    if request.method == "POST":
        job_desc = request.form.get("job_desc")
        uploaded_files = request.files.getlist("resumes")

        if not job_desc or not uploaded_files:
            error = "Please upload resumes and enter a job description."
            return render_template("index.html", error=error)

        jd_embedding = model.encode(job_desc)
        threshold = 0.4  # lower to allow more flexibility

        for uploaded_file in uploaded_files:
            filename = uploaded_file.filename
            try:
                resume_text = extract_text(uploaded_file)
                resume_embedding = model.encode(resume_text)
                similarity = util.cos_sim(resume_embedding, jd_embedding)[0][0].item()

                matched_skills = extract_skills(resume_text)
                skill_match_ratio = len(matched_skills) / len(COMMON_SKILLS)

                is_match = similarity >= threshold or skill_match_ratio >= 0.3

                results[filename] = {
                    "match": is_match,
                    "skills": matched_skills,
                    "similarity": round(similarity, 2),
                }
            except Exception as e:
                results[filename] = {
                    "match": False,
                    "skills": [],
                    "similarity": 0.0,
                }

        return render_template("index.html", results=results)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)

