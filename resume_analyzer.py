import streamlit as st
import os
import sys
import tempfile
import shutil
import base64
import uuid
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document

load_dotenv()


def display_pdf_preview(pdf_file):
    """Display PDF preview in sidebar."""
    try:
        st.sidebar.subheader("Resume Preview")
        base64_pdf = base64.b64encode(pdf_file.getvalue()).decode("utf-8")
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="500"></iframe>'
        st.sidebar.markdown(pdf_display, unsafe_allow_html=True)
    except Exception as e:
        st.sidebar.error(f"Error previewing PDF: {str(e)}")


def extract_text_from_pdf(uploaded_file):
    """Extract text content from uploaded PDF file."""
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text


def run_rag_completion(docs, query_text, job_title, job_description, llm, embeddings):
    """Run RAG pipeline using FAISS + Groq + HuggingFace embeddings."""

    
    vectorstore = FAISS.from_documents(docs, embeddings)

    
    retriever = vectorstore.as_retriever(search_type="similarity", k=5)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False,
    )

    # Compact, clear prompt for LLM
    prompt = f"""
    You are a professional resume coach. 
    Improve the resume for the job below.

    Job Title: {job_title}
    Job Description: {job_description}

    Task: {query_text}

    Respond with:
    ## Key Findings
    • Main matches and gaps

    ## Improvements
    • 3–5 strong, actionable improvements

    ## Action Items
    • 2–3 immediate next steps
    """

    
    response = qa_chain.invoke(
        {"query": prompt},
        config={"configurable": {"run_id": str(uuid.uuid4())}}
    )
    return response["result"]


#  Main App 
def main():
    st.set_page_config(page_title="Resume Optimizer", layout="wide")

    # Session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.title("📝 AI Resume Optimizer")
    st.caption("Powered by Groq + HuggingFace + FAISS")

    
    with st.sidebar:
        st.subheader("🔑 API Keys")
        groq_api = st.text_input("Enter Groq API Key", type="password")
        hf_token = st.text_input("Enter HuggingFace Token", type="password")

        st.divider()
        uploaded_file = st.file_uploader("📄 Upload Resume (PDF)", type="pdf")

        if uploaded_file:
            display_pdf_preview(uploaded_file)

    # Main UI
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Job Information")
        job_title = st.text_input("Job Title")
        job_description = st.text_area("Job Description", height=200)

        st.subheader("Optimization Focus")
        optimization_type = st.selectbox(
            "Choose optimization type",
            [
                "ATS Keyword Optimizer",
                "Experience Enhancer",
                "Skills Hierarchy",
                "Professional Summary",
                "Education Optimizer",
                "Technical Skills",
                "Career Gap Handling",
            ],
        )

        if st.button("🚀 Optimize Resume"):
            if not uploaded_file:
                st.error("Please upload your resume first")
                st.stop()
            if not groq_api or not hf_token:
                st.error("Please enter both API keys")
                st.stop()

            # Extract resume text
            resume_text = extract_text_from_pdf(uploaded_file)
            docs = [Document(page_content=resume_text)]

            # Init embeddings + LLM
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
            )

            llm = ChatGroq(
                groq_api_key=groq_api,
                model_name="llama-3.1-8b-instant",  
                temperature=0.3,
            )

            
            prompts = {
                "ATS Keyword Optimizer": "Identify missing ATS keywords and suggest improvements.",
                "Experience Enhancer": "Enhance experience section with measurable achievements.",
                "Skills Hierarchy": "Organize and rank skills as per job requirements.",
                "Professional Summary": "Write a strong professional summary tailored for the job.",
                "Education Optimizer": "Highlight education relevant to this job.",
                "Technical Skills": "Optimize technical skills section.",
                "Career Gap Handling": "Frame career gaps positively and professionally.",
            }

            with st.spinner("Analyzing resume..."):
                try:
                    response = run_rag_completion(
                        docs,
                        prompts[optimization_type],
                        job_title,
                        job_description,
                        llm,
                        embeddings,
                    )
                    response = response.replace("<think>", "").replace("</think>", "")
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response}
                    )
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    with col2:
        st.subheader("Optimization Results")
        if st.session_state.messages:
            for message in st.session_state.messages:
                st.markdown(message["content"])


if __name__ == "__main__":
    main()

