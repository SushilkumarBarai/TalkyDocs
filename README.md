# 📘 TalkyDocs: Smart PDF QnA System (RAG-powered)

TalkyDocs is a Streamlit-based Question & Answer system that enables users to upload any PDF document and instantly get accurate answers by asking natural language questions. Powered by Retrieval-Augmented Generation (RAG), Ollama LLMs, ChromaDB vector search, and reranking with Sentence Transformers, TalkyDocs delivers high-quality, context-aware responses in real time.

---

## 🚀 Features

- 📄 Upload and process PDF documents
- 🔍 Document chunking and vector storage using ChromaDB
- 🧠 Embedding powered by Ollama’s `nomic-embed-text` model
- ⚙️ Cosine similarity search for semantic relevance
- 📚 Cross-encoder re-ranking using MS MARCO MiniLM
- 🧾 Context-aware LLM responses via `llama3.2:3b`
- 📡 Real-time streaming of model responses
- 🖼️ Simple, beautiful, and interactive Streamlit interface

---

## 🎯 Use Cases

- 💰 **Finance**: Ask questions over financial reports, regulatory filings, and investment summaries.
- ⚖️ **Legal**: Instantly extract insights from contracts, case laws, and legal policies.
- 🏥 **Healthcare**: Query clinical papers, patient records, or drug research efficiently.
- 🎓 **Academia**: Summarize research papers and get answers from academic PDFs.
- 🧑‍💼 **HR & Training**: Retrieve information from policy documents and onboarding materials.

--

## 🛠️ Tech Stack

- **Streamlit** – UI framework.
- **ChromaDB** – Persistent vector store.
- **Ollama** – Local LLM server for inference & embedding.
- **LangChain** – Document loading and splitting.
- **Sentence Transformers** – Cross-encoder for reranking.
- **PyMuPDF** – PDF parsing and reading.

---

## Screenshot 

Here’s a screenshot of the application:

### Sample 1


![Screenshot_1](https://github.com/SushilkumarBarai/TalkyDocs/blob/main/images/Screenshot_1.png)


### Sample 2

![Screenshot_2](https://github.com/SushilkumarBarai/TalkyDocs/blob/main/images/Screenshot_1.png)


### Sample 3

![Screenshot_2](https://github.com/SushilkumarBarai/TalkyDocs/blob/main/images/Screenshot_3.png)



## 📦 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/SushilkumarBarai/TalkyDocs.git
cd TalkyDocs
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Start Ollama and Pull Required Models

```bash
ollama run nomic-embed-text
ollama run llama3.2:3b
```

### 4. Launch TalkyDocs

```bash
streamlit run app.py
```

