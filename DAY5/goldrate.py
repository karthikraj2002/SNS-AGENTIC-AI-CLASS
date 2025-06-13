import streamlit as st
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.utilities import DuckDuckGoSearchAPIWrapper
import time
from typing import Optional

# 🎯 Gemini API key (use env variables in production)
GOOGLE_API_KEY = "AIzaSyCRLwj_8icu9TSghzriDzlAqluGGjokLZ8"

def create_agent_with_retry() -> Optional[object]:
    """Create LangChain agent with Gemini + DuckDuckGo"""
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.3,
            max_tokens=1000,
            request_timeout=30
        )

        search = DuckDuckGoSearchAPIWrapper()
        search_tool = Tool(
            name="DuckDuckGo Search",
            func=search.run,
            description="Useful for answering questions about current events or recent facts from the internet."
        )

        agent = initialize_agent(
            tools=[search_tool],
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,
            max_iterations=3,
            handle_parsing_errors=True
        )
        return agent

    except Exception:
        return None

def query_with_backoff(agent, query: str, max_retries: int = 3) -> str:
    """Query agent with retry logic for rate limits"""
    for attempt in range(max_retries):
        try:
            return agent.run(query)
        except Exception as e:
            error_msg = str(e).lower()

            if "429" in error_msg or "resource" in error_msg or "quota" in error_msg:
                if attempt < max_retries - 1:
                    time.sleep((2 ** attempt) * 5)
                    continue
                else:
                    return "❌ **API Quota Exceeded**: Your Gemini API key has reached its usage limit. Try again later."
            elif "timeout" in error_msg:
                return "⏰ **Request Timeout**: That took too long. Try a simpler question."
            else:
                return f"⚠️ **Error**: {str(e)[:200]}... Try again."

    return "❌ **Max Retries Exceeded**: Unable to process request."

# 🌐 Streamlit UI
st.set_page_config(page_title="Live Q&A with Gemini 🌐", page_icon="🔍")
st.title("🌍 Ask Anything Live! 🔍")
st.write("Powered by **Gemini** + **DuckDuckGo** with LangChain Agent 🧠")

with st.expander("ℹ️ Usage Information"):
    st.write("""
    **Tips:**
    - Ask clear, specific questions
    - Avoid long multi-part queries

    **Common Errors:**
    - **Quota Exceeded (429)**: Try later, or get a new API key
    - **Timeout**: Ask simpler question
    """)

# Initialize agent
if 'agent' not in st.session_state:
    with st.spinner("Initializing AI agent... 🤖"):
        st.session_state.agent = create_agent_with_retry()

query = st.text_input("💬 Type your question about current events or facts:")
submit = st.button("Ask 🤖")

if 'last_query_time' not in st.session_state:
    st.session_state.last_query_time = 0

if submit and query:
    if st.session_state.agent is None:
        st.error("❌ Failed to initialize agent. Please check your API key and refresh.")
    else:
        current_time = time.time()
        wait_time = current_time - st.session_state.last_query_time

        if wait_time < 3:
            st.warning(f"⏳ Please wait {3 - int(wait_time)} more seconds.")
        else:
            with st.spinner("Thinking... 🤔"):
                response = query_with_backoff(st.session_state.agent, query)

            st.session_state.last_query_time = current_time

            if response.startswith(("❌", "⚠️", "⏰")):
                st.error(response)
            else:
                st.success("✅ Answer:")
                st.write(response)

if st.button("🔄 Reset Agent"):
    st.session_state.agent = None
    st.rerun()
