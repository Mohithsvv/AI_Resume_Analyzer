import os
from google import genai
from dotenv import load_dotenv
import streamlit as st

st.title("AI_Resume_Builder")

google_api_key = st.secrets["GOOGLE_API_KEY"]
if not google_api_key:
    st.error("Please check your google api keys")
    st.stop()

client=genai.Client(api_key=google_api_key)

resume_text=st.text_area("Resume"," ",height=200)
job_text=st.text_area("Job Description"," ",height=300)

model_text=st.selectbox("Model",["gemini-2.0-flash","gemini-2.5-flash","gemini-1.5-flash"])
temp=st.slider("Temperature",0.0,1.0,0.7)
max_tokens=st.number_input("Max Tokens",min_value=64,max_value=2000,value=800)

def extract_text(resp):
    if hasattr(resp,"text") or resp.text:
        return resp.text
    if hasattr(resp,"output_text") or resp.output_text:
        return resp.output_text
    try:
        return str(resp)
    except Exception:
        return "<no text>"

def build_prompt(resume,job):
    return f"""
Context:
    You are a professional career advisor and resume writer. Analyze the candidate's resume against the provided job description to assess alignment and identify areas for improvement. 

    Tasks:
    1. Summarize key requirements from the Job Description.  
    2. Highlight relevant experience from the Resume that matches those requirements.  
    3. Identify gaps or missing skills, especially in AI areas (Prompt Engineering, ChatGPT, GenAI, Agentic AI).  
    4. Highlight additional strengths from the Resume that add value.  
    5. Rewrite and enhance the Resume tailored for this job, ensuring:  
    - Suitability Score (out of 100) 
    - Strong action verbs and quantifiable results.  
    - Emphasis on relevant skills/experience.  
    - AI-related skills (Prompt engineering, ChatGPT, LLMs, Agentic AI) are clearly showcased.  
    - Resume is well-formatted and impactful.

Input:
Resume:
{resume}

Job Description:
{job}

Output:
    - Analysis (sections 1–4 as above).  
    - Updated, improved resume.
"""

if st.button("Generate Improvised Resume"):
    if not resume_text.strip() or not job_text.strip():
        st.warning("Enter both resume and job description too")
        
    else:
        prompt = build_prompt(resume_text,job_text)
        with st.spinner("Analysising Resume and Rewriting Resume"):
            try:
                resp=client.models.generate_content(
                    model=model_text,
                    contents=[prompt]
                )
                out=extract_text(resp)

            except Exception as e:
                out = f"Gemini is giving {e}"
        st.subheader("Improved Resume")
        st.markdown(out)
