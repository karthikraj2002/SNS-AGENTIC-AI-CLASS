import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.messages import AIMessage, HumanMessage

# ---- SET YOUR GOOGLE GEMINI API KEY HERE ----
GEMINI_API_KEY = "AIzaSyCWjDvfquO7Yf6cOrnmC9xNE-1X_jVZiaM"

# ---- Initialize LLM using LangChain ----
try:
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GEMINI_API_KEY)
except Exception as e:
    st.error(f"Failed to initialize Gemini: {e}")
    st.stop()

# ---- Prompt Template for Translation ----
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that translates English to French."),
    ("human", "Translate the following sentence to French:\n\n{english_sentence}")
])

# ---- Chain (prompt | llm) ----
translation_chain: Runnable = prompt | llm

# ---- Streamlit UI ----
st.set_page_config(page_title="English to French Translator", page_icon="🌍")
st.title("🌍 English to French Translator")
st.markdown("Enter an English sentence and click **Translate** to get the French version.")

# ---- User Input ----
english_input = st.text_input("Enter English Sentence:", placeholder="e.g., How are you?")

if st.button("Translate"):
    if not english_input.strip():
        st.warning("Please enter a sentence before clicking Translate.")
    else:
        try:
            # Run the chain
            response = translation_chain.invoke({"english_sentence": english_input})

            # Extract and display response
            if isinstance(response, AIMessage):
                french_output = response.content
            else:
                french_output = str(response)

            st.success("✅ Translation successful!")
            st.text_area("French Translation:", value=french_output, height=150)
        except Exception as e:
            st.error(f"❌ Error during translation: {e}")
