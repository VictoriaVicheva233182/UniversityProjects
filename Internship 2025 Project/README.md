# 🏗️ Construction Document Intelligence System
## AI-Powered RAG Solution for Technical Document Analysis

<div align="center">


**Local RAG system enabling natural language queries across technical construction documents**  
*Built during a 20-week software engineering internship at a Dutch steel construction company*

[Overview](#-overview) • [Features](#-features) • [Architecture](#-architecture) • [Demo](#-demo) • [Technical Details](#-technical-details)

</div>

---

## 📋 Overview

This project demonstrates a production-ready **Retrieval-Augmented Generation (RAG)** system designed for the construction industry. It enables technical teams to instantly find information across thousands of pages of specifications, calculations, safety protocols, and project documentation using natural language queries in multiple languages.

### 🎯 The Problem

Construction and engineering companies manage vast libraries of technical documents:
- Structural calculations spanning hundreds of pages
- Material specifications across dozens of projects
- Safety protocols and compliance documentation
- Project meeting notes and correspondence
- Technical standards and building codes

**Finding specific information traditionally requires:**
- Manual searching through multiple PDF files
- Remembering which document contains what information
- Understanding technical terminology in multiple languages
- Cross-referencing between related documents

### 💡 The Solution

An intelligent document analysis system that:
- ✅ **Understands natural language questions** in Dutch and English
- ✅ **Searches across all documents simultaneously** using semantic understanding
- ✅ **Provides accurate answers with source citations** for verification
- ✅ **Processes multiple file formats** (PDF, Word, Excel) with OCR support
- ✅ **Runs completely locally** for data security and privacy
- ✅ **Scales to millions of document chunks** without performance degradation

### ✨ Key Achievements

- **Production deployment:** Successfully scaled from 10K prototype to 1.5M+ document chunks
- **Multi-format processing:** Handles PDF (with OCR), DOCX, XLSX seamlessly
- **Multilingual capability:** Automatic language detection and response in Dutch/English
- **Enterprise features:** Multi-user authentication, session management, professional UI
- **Privacy-first:** 100% local deployment with zero external API calls
- **Empirical optimization:** Model selection based on comparative testing (Mistral 7B vs Qwen2 vs Llama 3.1)

---

## 🚀 Features

### Core Capabilities

**🤖 Intelligent Query Processing**
- Natural language understanding in Dutch and English
- Context-aware responses with anti-hallucination measures
- Source citations for every claim
- Powered by locally-hosted Mistral 7B LLM

**📄 Comprehensive Document Support**
- **PDF:** Full text extraction with OCR for scanned documents (Tesseract)
- **Microsoft Word:** Complete .docx parsing including tables and formatting
- **Excel Spreadsheets:** Multi-sheet processing with cell-level extraction
- **Automatic language detection** for optimal processing

**🔍 Advanced Retrieval System**
- FAISS vector database for sub-100ms similarity search
- Multilingual embeddings (paraphrase-multilingual-mpnet-base-v2)
- Semantic search across 1.5M+ document chunks
- Configurable retrieval parameters (chunk size, overlap, top-k)

**👥 Enterprise-Ready Features**
- Secure JWT-based authentication
- Multi-user support with session management
- Role-based access control framework
- Professional, branded user interface
- RESTful API with comprehensive documentation

**🐳 Modern DevOps**
- Fully containerized with Docker Compose
- One-command deployment
- Persistent data volumes
- Health monitoring endpoints
- Production-ready configuration

---

## 🏗️ Architecture

### Technology Stack

| Layer                   | Technologies                                                  |
| ----------------------- | ------------------------------------------------------------- |
| **Frontend**            | React 18.2, Vite 5.0, Tailwind CSS, Axios, React Router       |
| **Backend**             | FastAPI 0.104, Python 3.10, Uvicorn ASGI server               |
| **LLM**                 | Ollama 0.12.6, Mistral 7B Instruct (4.5GB)                    |
| **Vector DB**           | FAISS (Facebook AI Similarity Search)                         |
| **Embeddings**          | Sentence-Transformers (paraphrase-multilingual-mpnet-base-v2) |
| **OCR**                 | Tesseract 5.x (English, Dutch language packs)                 |
| **Document Processing** | PyPDF2, pdf2image, python-docx, openpyxl, Pillow              |
| **Authentication**      | python-jose (JWT), passlib, bcrypt                            |
| **Infrastructure**      | Docker 24.x, Docker Compose 2.x, Nginx (Alpine)               |

---

## 🎥 Demo

### Privacy & Access Notice

⚠️ **This system is fully deployed locally and contains sensitive company information.** 

Due to the local-only nature of this deployment and strict data confidentiality requirements, **installation and live access are restricted to authorized personnel within the organization only.**

### Demo Video

A demonstration video showcasing the system's capabilities is available for external viewing.

**📺 [Watch Demo Video](#)** *(Link to be added)*

---

## 💻 How It Works

### User Workflow

**1. Document Upload**
- Upload construction documents (PDF, DOCX, XLSX)
- Automatic OCR processing for scanned files
- Documents indexed in 30-60 seconds

**2. Natural Language Queries**
- Ask questions in Dutch or English
- System retrieves relevant context
- LLM generates accurate answers with citations

**3. Verify & Export**
- Review answers with source references
- Click citations to view original documents
- Export results for reports

### Sample Queries

**English:**
- "What are the safety requirements for working at height?"
- "Show me specifications for HEA 320 steel columns"
- "What is the maximum wind load for portal frames?"

**Dutch:**
- "Wat zijn de specificaties voor staalkolommen?"
- "Geef me de veiligheidsvereisten voor lassen"
- "Wat is de draagkracht van IPE 400 liggers?"

---

## 📊 Technical Details

### RAG Pipeline

**Document Ingestion:**
```
Document → Format Detection → OCR (if needed) → Text Extraction
    → Language Detection → Chunking (500 chars, 50 overlap)
    → Embedding Generation → FAISS Index Update
```

**Query Processing:**
```
User Query → Query Embedding → Similarity Search (top-k=5)
    → Context Assembly → Prompt Construction
    → LLM Generation → Answer + Citations
```

### Model Selection

Empirical evaluation of three 7B-8B parameter models:

| Model          | Performance | Speed  | Memory | Selected  |
| -------------- | ----------- | ------ | ------ | --------- |
| **Mistral 7B** | ⭐⭐⭐⭐⭐       | Fast   | 4.5GB  | ✅ **Yes** |
| Qwen2 7B       | ⭐⭐⭐⭐        | Medium | 4.8GB  | ❌ No      |
| Llama 3.1 8B   | ⭐⭐⭐⭐        | Slow   | 5.2GB  | ❌ No      |

**Mistral 7B selected for:**
- Best accuracy on technical terminology
- Fastest inference time
- Excellent multilingual performance

### Anti-Hallucination Measures

- Strict prompt engineering (answer only from context)
- Required source citations for all claims
- Low temperature (0.3) for factual responses
- Limited context window (top-5 chunks)

### Performance Metrics

**Production Performance:**
- Query response: < 2 seconds average
- Document processing: ~1 min per 50-page PDF
- Vector search: < 100ms for 1.5M vectors
- Concurrent users: Tested up to 10 simultaneous

**Accuracy (Internal Testing):**
- Retrieval precision: 92% (correct documents in top-5)
- Answer accuracy: 89% (verified against ground truth)
- Citation accuracy: 95% (correct page references)

---

## 📁 Project Structure
```
construction-doc-intelligence/
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   ├── core/
│   │   ├── llm_service.py
│   │   ├── vector_store.py
│   │   ├── document_processor.py
│   │   └── embedding_service.py
│   └── api/
│       ├── auth.py
│       ├── documents.py
│       └── chat.py
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/
│   │   └── services/
│   ├── package.json
│   └── vite.config.js
├── docker-compose.yml
├── Dockerfile
└── README.md
```

---

## 🎓 Project Context

This system was developed during a **20-week software engineering internship** at De Kok Staalbouw, a Dutch steel construction company, for their Calculations Department.

**Key Project Phases:**

1. **Research & Planning (Weeks 1-3)** - Requirements gathering, technology evaluation
2. **Prototype Development (Weeks 4-8)** - Core RAG pipeline, 10K document proof-of-concept
3. **Production Scaling (Weeks 9-15)** - Multi-format support, scaled to 1.5M+ chunks
4. **Deployment & Refinement (Weeks 16-20)** - Docker containerization, user testing

**Academic Deliverables:**
- ✅ Production-ready software system
- ✅ Technical documentation
- ✅ Ethics analysis report
- ✅ Literature review
- ✅ Portfolio documentation

---

## 🔐 Security & Privacy

### Data Confidentiality

- **100% local deployment** - No cloud services, no external APIs
- **Zero data exposure** - Documents never leave company premises
- **No internet dependency** - Operates in air-gapped environments
- **Complete control** - Organization owns all data and infrastructure

### Authentication & Authorization

- JWT-based authentication with configurable expiration
- Password hashing using bcrypt (cost factor 12)
- Session management with automatic cleanup
- Role-based access control framework

---

## 🚧 Known Limitations

- **Hardware requirements:** Requires decent CPU/RAM (Mistral 7B needs ~4.5GB)
- **Processing time:** Large documents (>100 pages) take 2-3 minutes
- **Language support:** Currently limited to Dutch and English
- **File size cap:** 50MB per document (configurable)
- **No real-time collaboration:** Single-user sessions only

---

## 🛣️ Roadmap

**Completed:**
- ✅ Core RAG pipeline
- ✅ Multi-format document processing
- ✅ Multi-user authentication
- ✅ Docker containerization
- ✅ Production deployment

**In Progress:**
- 🔄 System stability optimization
- 🔄 Advanced analytics dashboard

**Future Enhancements:**
- ⏳ Additional language support (German, French)
- ⏳ Integration with CAD software
- ⏳ Mobile application

---

## 🤝 Contact

**Victoria Vicheva**  
- GitHub: [@s01-VictoriaVicheva233182](https://github.com/VictoriaVicheva233182/UniversityProjects.git)
- LinkedIn: [Your LinkedIn](www.linkedin.com/in/victoria-vicheva-3817b6263)
- Email: [victoria.v.vicheva@gmail.com]

---

## 🙏 Acknowledgments

- **De Kok Staalbouw** for the internship opportunity
- **Uther Tlas** university supervision
- **Calculations Department** for testing and feedback
- **Breda University of Applied Sciences** for academic support

---

<div align="center">

**Built with ❤️ during internship at De Kok Staalbouw**

*Empowering construction professionals with AI-powered document intelligence*

</div>