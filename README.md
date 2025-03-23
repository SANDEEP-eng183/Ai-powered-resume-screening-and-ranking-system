# Ai-powered-resume-screening-and-ranking-system
The system Uses AI to automate the resume screening processing. Ranking candidate based on the qualifications and skills and fit for the job. 
import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import hashlib
import smtplib
import matplotlib.pyplot as plt
from email.mime.text import MIMEText

# Function to extract structured details (Name, Email, Phone, Skills) from resume
def extract_resume_details(text):
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b'
    phone_pattern = r'\b\d{10}\b'
    
    email = re.findall(email_pattern, text)
    phone = re.findall(phone_pattern, text)
    
    # Extract skills based on common tech keywords
    skills = [word for word in text.split() if word.lower() in {'python', 'java', 'sql', 'machine learning', 'cloud', 'aws', 'data science'}]
    
    return email[0] if email else "Not Found", phone[0] if phone else "Not Found", ', '.join(set(skills))

# Function to hash resume text for duplicate detection
def hash_resume(text):
    return hashlib.md5(text.encode()).hexdigest()

# Function to rank resumes based on AI scoring
def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()
    
    job_desc_vector = vectors[0]
    resume_vectors = vectors[1:]
    similarity_scores = cosine_similarity([job_desc_vector], resume_vectors).flatten()

    return similarity_scores * 100  # Normalize to percentage

# Function to send email (for shortlisted candidates)
def send_email(candidate_email, status):
    smtp_server = "smtp.example.com"
    sender_email = "hr@example.com"
    sender_password = "your_password"

    subject = "Interview Update"
    body = f"Dear Candidate,\n\nWe are pleased to inform you that you have been {status} for the next round.\n\nBest Regards,\nHR Team"

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = candidate_email

    try:
        server = smtplib.SMTP(smtp_server, 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, candidate_email, msg.as_string())
        server.quit()
        return True
    except:
        return False

# Streamlit UI
st.title("🔍 AI-Powered Resume Screening System for HR Teams")

# Job description input
st.header("📌 Job Description")
job_description = st.text_area("Enter the job description", height=150)

# Resume upload section
st.header("📂 Upload Resumes")
uploaded_files = st.file_uploader("Upload Resumes (PDF)", type=["pdf"], accept_multiple_files=True)

# Sorting option
sort_order = st.radio("📊 Sort Order", ["Descending (Best to Worst)", "Ascending (Worst to Best)"])

if uploaded_files and job_description:
    st.header("📊 Processing Resumes...")

    resume_data = []
    hashes = set()  # For duplicate detection

    for file in uploaded_files:
        pdf = PdfReader(file)
        text = " ".join([page.extract_text() for page in pdf.pages if page.extract_text()])
        
        resume_hash = hash_resume(text)
        if resume_hash in hashes:
            st.warning(f"⚠️ Duplicate Resume Found: {file.name}. Skipping...")
            continue
        hashes.add(resume_hash)

        email, phone, skills = extract_resume_details(text)
        resume_data.append({"File Name": file.name, "Email": email, "Phone": phone, "Skills": skills, "Text": text})

    scores = rank_resumes(job_description, [r["Text"] for r in resume_data])

    # Create a DataFrame with scores
    results = pd.DataFrame(resume_data)
    results["Score"] = scores
    results = results.sort_values(by="Score", ascending=(sort_order == "Ascending (Worst to Best)"))

    # Display results
    st.dataframe(results[["File Name", "Score", "Email", "Phone", "Skills"]])

    # Shortlist candidates
    threshold = st.slider("📈 Shortlisting Score Threshold", 0, 100, 70)
    shortlisted = results[results["Score"] >= threshold]

    st.subheader(f"✅ Shortlisted Candidates ({len(shortlisted)})")
    st.dataframe(shortlisted[["File Name", "Score", "Email"]])

    # Send Interview Emails
    if st.button("📧 Send Interview Emails"):
        for _, row in shortlisted.iterrows():
            if send_email(row["Email"], "shortlisted"):
                st.success(f"✅ Email sent to {row['Email']}")
            else:
                st.error(f"❌ Failed to send email to {row['Email']}")

    # Graphical Analysis
    st.subheader("📊 Resume Score Distribution")
    fig, ax = plt.subplots()
    ax.bar(results["File Name"], results["Score"], color='blue')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.xlabel("Candidates")
    plt.ylabel("Match Score")
    plt.title("Resume Ranking")
    st.pyplot(fig)

    # Download CSV
    csv = results.to_csv(index=False)
    st.download_button("📥 Download Full Report", csv, "resume_analysis.csv", "text/csv")
