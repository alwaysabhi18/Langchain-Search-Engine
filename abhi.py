import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the tools for querying
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1000)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=1000)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

search = DuckDuckGoSearchRun(name="Search")

st.title("🔎 LangChain - Search Engine")

# Sidebar for settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]

# Display chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"**User:** {msg['content']}")
    else:
        st.markdown(f"**Assistant:** {msg['content']}")

# User input for query
prompt = st.text_input("Ask something:", placeholder="What is machine learning?")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(f"**User:** {prompt}")

    # Initialize LLM (Large Language Model)
    llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192", streaming=True)
    tools = [wiki, arxiv, search]

    # Initialize agent for processing queries
    search_agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True)

    # Process the query with the agent
    with st.spinner("Gathering information..."):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        
        # Get responses from all tools
        wiki_response = search_agent.run(prompt, callbacks=[st_cb], tools=[wiki])
        arxiv_response = search_agent.run(prompt, callbacks=[st_cb], tools=[arxiv])
        search_response = search_agent.run(prompt, callbacks=[st_cb], tools=[search])
        
        # Combine all responses
        combined_response = f"**Wikipedia:** {wiki_response}\n\n" \
                            f"**Arxiv:** {arxiv_response}\n\n" \
                            f"**DuckDuckGo:** {search_response}"
        
        # Append the response to chat history
        st.session_state.messages.append({'role': 'assistant', "content": combined_response})
        
        # Format and display the response
        formatted_response = combined_response.replace('\n', '<br>')
        st.markdown(f"**Assistant:** {formatted_response}", unsafe_allow_html=True)
