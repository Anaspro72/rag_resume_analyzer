# 📝 AI Resume Optimizer

An interactive **Streamlit web app** that helps you **analyze and optimize your resume** for a specific job.  
Powered by **Groq LLMs**, **Hugging Face embeddings**, and **FAISS vector search**, the app provides actionable insights to improve your resume.

---

## 🚀 Features
- Upload your resume in **PDF** format and preview it directly in the app.
- Enter a **Job Title** and **Job Description** to customize recommendations.
- Choose different optimization modes:
  - ✅ ATS Keyword Optimizer  
  - ✅ Experience Enhancer  
  - ✅ Skills Hierarchy  
  - ✅ Professional Summary  
  - ✅ Education Optimizer  
  - ✅ Technical Skills  
  - ✅ Career Gap Handling
- Get **tailored improvement suggestions** and **action items**.
- Lightweight and deployable on **Streamlit Cloud**.



## 🔑 API Keys Required
This app needs two API keys:

### 1. Groq API Key
- Sign up at [Groq Cloud](https://console.groq.com/keys).
- Create a new API key from the **API Keys** section.
- Copy the key (it starts with `gsk_...`).

### 2. Hugging Face Token
- Sign up at [Hugging Face](https://huggingface.co/).
- Go to **Settings > Access Tokens**.
- Create a new token (read access is enough).
- Copy the token (it starts with `hf_...`).

You’ll enter both keys in the sidebar of the app.

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/your-username/resume-optimizer.git
cd resume-optimizer
