# AI Medical Research Assistant 🔬

This project is a sophisticated, AI-powered chatbot designed to make complex medical research accessible and understandable. Users can ask questions in natural language and receive answers grounded in real medical literature from PubMed, general web knowledge, and a self-improving memory. The application is built with a modern agentic architecture, a MongoDB Atlas vector database, and an interactive Streamlit UI.



---
## Key Features

* **Retrieval-Augmented Generation (RAG):** Answers are grounded in factual data ingested from PubMed, minimizing AI hallucinations and providing trustworthy information.
* **Multi-Tool Agent:** The AI agent intelligently chooses the best tool for a user's query:
    * 🧠 **Internal Knowledge:** Performs a vector search on ingested PubMed abstracts for deep, scientific questions.
    * 🌐 **Web Search:** Searches the web for general knowledge, news, and topics not covered in its internal database.
    * 📖 **Term Definer:** Provides simple, on-the-fly definitions of complex medical jargon.
* **Self-Improving Memory:** The chatbot learns from user feedback. Positive interactions create "shortcuts" in a `learned_associations` database, allowing it to answer similar questions faster and more efficiently over time.
* **Interactive UI:** A user-friendly chat interface built with Streamlit that shows the agent's status and allows for easy feedback.

---
## Architecture: RAG with MongoDB Atlas

This project's intelligence is built on a **Retrieval-Augmented Generation (RAG)** architecture. RAG makes our AI more reliable by forcing it to answer questions based on a trusted knowledge base, like an "open-book exam," rather than relying solely on its internal memory. This dramatically reduces the risk of the AI making up incorrect information.



### Why MongoDB Atlas is the Ideal Backend for RAG

MongoDB Atlas serves as the perfect data backbone for this architecture, combining the roles of an operational database and a high-performance vector database in one unified platform.

* **Unified Data Platform:** Atlas isn't just a vector database. It's a flexible document database that allows us to store vector embeddings right alongside the original content, metadata (like sources and titles), user profiles, and chat history in a single JSON document. This eliminates the need to manage and sync separate databases, resulting in a simpler, more cost-effective system.

* **Atlas Vector Search:** This is the core engine that enables the "Retrieval" step of RAG.
    * **Semantic Understanding:** It goes beyond keywords to understand the *meaning* behind a query. It finds documents that are conceptually similar, not just textually identical.
    * **Speed and Scale:** Built for high performance, it can find the most relevant information in milliseconds, even across millions of documents.
    * **Hybrid Search Power:** Because it's integrated, we can combine vector search with traditional filters in a single, powerful query. For example: "Find articles *semantically similar* to 'diabetes treatments,' but only those *published after 2023* containing the *keyword 'metformin'*."

---
## Technology Stack

* **Database:** **MongoDB Atlas** (Operational Data, Vector Database, Learned Associations)
* **AI Models:** **Google AI Platform** (Gemini Pro, Gemini 1.5 Flash, Text Embedding Model)
* **Orchestration:** **LangChain**
* **UI Framework:** **Streamlit**
* **Data Sourcing:** **PubMed API** (`BioPython`), **Google Search** (`googlesearch-python`)

---
## Setup and Installation

Follow these steps to set up and run the project locally.

### 1. Prerequisites

* Python 3.9+
* A MongoDB Atlas account (the free M0 tier is sufficient).

### 2. Clone and Install Dependencies

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
# Install the required libraries
pip install -r requirements.txt

### 3. Configure Environment Variables

Create a `.env` file in the root of your project folder and add your secret keys:

```env
MONGO_URI="mongodb+srv://<user>:<password>@your_cluster_url/..."
GOOGLE_API_KEY="your-google-api-key"
NCBI_API_KEY="your-ncbi-api-key"
```
### 4. Set Up MongoDB Atlas

1.  **IP Access:** In your Atlas dashboard, navigate to **Network Access** and ensure your current IP address is whitelisted.
2.  **Create Vector Indexes:** You must create two separate vector search indexes with the correct JSON configurations:
    * On the `medical_articles` collection, named `vector_index_medical`.
    * On the `learned_associations` collection, named `vector_index_associations`.

---
## Usage

### 1. Ingest Data

Before running the chatbot for the first time, you must populate its knowledge base.

```bash
python ingest_pubmed.py
```
### 2. Run the Chatbot Application

To start the interactive web UI, run the following command:
```bash
streamlit run app.py
```
