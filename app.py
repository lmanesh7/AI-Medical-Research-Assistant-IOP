# file: app.py (Updated with UI notifications)

import streamlit as st
import os
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools.retriever import create_retriever_tool
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from googlesearch import search as google_search_func
from langchain.tools import tool

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Medical Research Assistant",
    page_icon="🔬",
    layout="centered"
)

# --- Backend Setup ---
@st.cache_resource
def load_backend():
    print("Loading backend resources...")
    load_dotenv()
    
    agent_llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0, convert_system_message_to_human=True)
    framing_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7, convert_system_message_to_human=True)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    client = MongoClient(os.environ["MONGO_URI"])
    db = client.medicalDB
    
    medical_collection = db.medical_articles_iop
    vector_store = MongoDBAtlasVectorSearch(collection=medical_collection, embedding=embeddings, index_name="vector_index_medical")
    retriever = vector_store.as_retriever()
    retriever_tool = create_retriever_tool(retriever, "medical_research_search", "Use for deep scientific questions about medical research.")
    
    @tool
    def web_search(query: str) -> str:
        """Use for general knowledge questions, current events, or topics not found in the medical research database."""
        try:
            return " ".join(list(google_search_func(query)))
        except Exception as e:
            return f"Error during web search: {e}"

    @tool
    def define_medical_term(term: str) -> str:
        """Use this tool to get a simple definition of a complex medical term."""
        definer_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
        prompt = f"Explain the medical term '{term}' in simple, easy-to-understand sentences."
        return definer_llm.invoke(prompt).content

    tools = [retriever_tool, web_search, define_medical_term]
    
    agent_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an advanced AI medical assistant. Choose the best tool for the user's question. Output only the direct result from the tool call."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    agent = create_tool_calling_agent(agent_llm, tools, agent_prompt)
    
    # --- MODIFIED LINE ---
    # We tell the agent to return its internal steps
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)
    
    framing_prompt_template = """You are a helpful AI assistant explaining complex topics simply. Frame the raw information below into a clear, conversational answer. If it's from medical research, add a disclaimer: "As an AI assistant, I am not a medical professional. This information is for educational purposes only. Please consult a doctor for medical advice."

    Original Question: {question}
    Raw Information: {context}
    
    Your Conversational Answer:"""
    framing_prompt = ChatPromptTemplate.from_template(framing_prompt_template)
    framing_chain = RunnablePassthrough() | framing_prompt | framing_llm | StrOutputParser()
    
    return agent_executor, framing_chain, db, embeddings, retriever

# --- Main App Logic ---
st.title("🔬 AI Medical Research Assistant")
st.subheader("IOP and Glaucoma")

agent_executor, framing_chain, db, embeddings, retriever = load_backend()
associations_collection = db.learned_associations
SIMILARITY_THRESHOLD = 0.92

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you today?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about a medical topic..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        final_answer = ""
        with st.spinner("Thinking..."):
            question_embedding = embeddings.embed_query(prompt)
            pipeline = [
                {"$vectorSearch": {"index": "vector_index_associations", "path": "question_embedding", "queryVector": question_embedding, "numCandidates": 5, "limit": 1}},
                {"$project": {"score": {"$meta": "vectorSearchScore"}, "answer_text": 1}}
            ]
            learned_results = list(associations_collection.aggregate(pipeline))
            
            if learned_results and learned_results[0]['score'] > SIMILARITY_THRESHOLD and False:
                st.info("Using a learned answer from memory...", icon="💡")
                raw_output = learned_results[0]['answer_text']
                final_answer = framing_chain.invoke({"question": prompt, "context": raw_output})
            else:
                agent_result = agent_executor.invoke({"input": prompt})
                raw_output = agent_result["output"]

                # --- NEW SECTION ---
                # Check which tool the agent used and display a notification
                if agent_result.get("intermediate_steps"):
                    tool_name = agent_result["intermediate_steps"][0][0].tool
                    if tool_name == "medical_research_search":
                        st.info("Searching ingested research articles...", icon="🧠")
                    elif tool_name == "web_search":
                        st.info("Searching the web for general information...", icon="🌐")
                    elif tool_name == "define_medical_term":
                        st.info("Defining medical term...", icon="📖")
                
                final_answer = framing_chain.invoke({"question": prompt, "context": raw_output})
            
            st.markdown(final_answer)
            st.session_state.last_response = {"question": prompt, "answer": final_answer}
    
    st.session_state.messages.append({"role": "assistant", "content": final_answer})


if len(st.session_state.messages) > 1 and "last_response" in st.session_state:
    last_question = st.session_state.last_response["question"]
    last_answer = st.session_state.last_response["answer"]

    st.write("---")
    st.markdown("<small>Was this answer helpful?</small>", unsafe_allow_html=True)
    
    col1, col2, _ = st.columns([1, 1, 5])
    
    if col1.button("👍 Yes", key=f"yes_{last_question}"):
        try:
            retrieved_docs = retriever.get_relevant_documents(last_question)
            if retrieved_docs:
                question_embedding = embeddings.embed_query(last_question)
                helpful_doc_id = retrieved_docs[0].metadata['_id']
                associations_collection.insert_one({
                    "question_embedding": question_embedding,
                    "helpful_doc_id": helpful_doc_id,
                    "question_text": last_question,
                    "answer_text": last_answer
                })
                st.toast("Thank you for your feedback! 👍")
        except Exception as e:
            st.error(f"Error saving feedback: {e}")

    if col2.button("👎 No", key=f"no_{last_question}"):
        st.toast("Thank you for your feedback. We'll use this to improve. 👎")