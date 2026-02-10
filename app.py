# file: app.py

import streamlit as st
import os
import time
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.retriever import create_retriever_tool
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from googlesearch import search as google_search_func
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="AI Medical Research Assistant",
    page_icon="🔬",
    layout="centered"
)

# --- 2. Robust Initialization Checks ---
def check_secrets():
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("Missing `GOOGLE_API_KEY`. Please set it in your .env file.")
        st.stop()
    if not os.getenv("MONGO_URI"):
        st.error("Missing `MONGO_URI`. Please set it in your .env file.")
        st.stop()

check_secrets()

# --- 3. Backend Setup (Cached) ---
@st.cache_resource(show_spinner="Initializing AI & Database...")
def load_backend():
    agent_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0, 
        max_retries=5,
        timeout=30
    )
    
    framing_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0.5
    )
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Mongo Connection
    try:
        client = MongoClient(os.environ["MONGO_URI"])
        db = client.medicalDB
        # Ping to ensure connection is alive
        client.admin.command('ping')
    except Exception as e:
        st.error(f"MongoDB Connection Failed: {e}")
        st.stop()
    
    medical_collection = db.medical_articles_iop
    vector_store = MongoDBAtlasVectorSearch(
        collection=medical_collection, 
        embedding=embeddings, 
        index_name="vector_index_medical"
    )
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 3}) # Limit to top 3 docs to save tokens
    
    retriever_tool = create_retriever_tool(
        retriever, 
        "medical_research_search", 
        "Use ONLY for deep scientific questions specifically about IOP, Glaucoma, and eye research."
    )
    
    @tool
    def web_search(query: str) -> str:
        """Use for general medical definitions, current events, or topics NOT found in the IOP database."""
        try:
            # Fetch 3 results to avoid rate limiting the Google Search library
            results = list(google_search_func(query, num=3, stop=3, pause=2.0))
            if not results:
                return "No web results found."
            return "Web Findings: " + " | ".join(results)
        except Exception as e:
            return f"Web search failed temporarily: {str(e)}"

    @tool
    def define_medical_term(term: str) -> str:
        """Use this tool to get a simple, layman definition of a complex medical term."""
        definer_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
        return definer_llm.invoke(f"Define {term} in one simple sentence.").content

    tools = [retriever_tool, web_search, define_medical_term]
    
    # Using MessagesPlaceholder is crucial for the agent to "see" the history
    agent_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert medical researcher. Always prioritize the 'medical_research_search' tool for IOP/Glaucoma questions. If the tool returns no data, fallback to web_search."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = create_tool_calling_agent(agent_llm, tools, agent_prompt)
    
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        return_intermediate_steps=True,
        handle_parsing_errors=True # Auto-fix formatting errors from the LLM
    )
    
    framing_prompt = ChatPromptTemplate.from_template(
        """You are a helpful medical assistant. 
        Summarize the following raw data into a friendly, clear response.
        
        Original User Question: {question}
        Raw Data from Tools: {context}
        
        If the data is medical, append this disclaimer: 
        *Disclaimer: I am an AI, not a doctor. Consult a specialist for medical advice.*
        """
    )
    
    framing_chain = framing_prompt | framing_llm | StrOutputParser()
    
    return agent_executor, framing_chain, db, embeddings, retriever

# --- 4. Load Resources ---
agent_executor, framing_chain, db, embeddings, retriever = load_backend()
associations_collection = db.learned_associations

# --- 5. Session State Management ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am ready to assist with your Glaucoma and IOP research."}]

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- 6. Sidebar Controls ---
with st.sidebar:
    st.header("Controls")
    if st.button("Clear Chat History", type="primary"):
        st.session_state.messages = [{"role": "assistant", "content": "Chat history cleared."}]
        st.session_state.chat_history = []
        st.rerun()

# --- 7. UI: Display Chat ---
st.title("🔬 AI Medical Research Assistant")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 8. UI: Input & Logic ---
if prompt := st.chat_input("Ask about IOP, medication, or studies..."):
    # Display user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing medical database..."):
            try:
                # A. Sliding Window Memory (Prevent Token Exhaustion)
                # We keep the last 6 turns (User + AI = 2 turns per interaction)
                memory_window = st.session_state.chat_history[-6:] if len(st.session_state.chat_history) > 6 else st.session_state.chat_history
                
                # B. Agent Execution
                agent_result = agent_executor.invoke({
                    "input": prompt,
                    "chat_history": memory_window
                })
                raw_output = agent_result["output"]
                
                # C. Final Framing
                final_answer = framing_chain.invoke({"question": prompt, "context": raw_output})
                
                st.markdown(final_answer)
                
                # D. Update State
                st.session_state.messages.append({"role": "assistant", "content": final_answer})
                st.session_state.chat_history.append(HumanMessage(content=prompt))
                st.session_state.chat_history.append(AIMessage(content=final_answer))
                
                # Save for feedback
                st.session_state.last_id = f"q_{int(time.time())}"
                st.session_state.last_response = {
                    "question": prompt, 
                    "answer": final_answer, 
                    "id": st.session_state.last_id
                }

            except Exception as e:
                st.error(f"An error occurred: {e}")
                # Log this error if you have a logging system

# --- 9. Robust Feedback System ---
# We check if a response exists to provide feedback on
if "last_response" in st.session_state:
    st.write("---")
    cols = st.columns([1, 1, 5])
    
    # We use a unique key per question ID to prevent button conflict
    qid = st.session_state.last_response["id"]
    
    if cols[0].button("👍 Helpful", key=f"up_{qid}"):
        try:
            last_q = st.session_state.last_response["question"]
            last_a = st.session_state.last_response["answer"]
            
            # Retrieve the doc ID that was actually relevant (if available)
            # For simplicity in this robust version, we re-retrieve the top hit
            docs = retriever.get_relevant_documents(last_q)
            doc_id = docs[0].metadata.get('_id', 'unknown') if docs else "web_search"
            
            associations_collection.insert_one({
                "timestamp": time.time(),
                "question": last_q,
                "answer": last_a,
                "feedback": "positive",
                "linked_doc_id": doc_id
            })
            st.toast("Feedback saved! We will learn from this.", icon="💾")
        except Exception as e:
            st.error(f"Database error: {e}")

    if cols[1].button("👎 Not Helpful", key=f"down_{qid}"):
        st.toast("Thanks for the feedback. We'll improve.", icon="📉")