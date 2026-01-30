import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import create_agent
from langchain_core.callbacks import BaseCallbackHandler

# -------------------------------
# Streamlit Callback for streaming
class StreamlitCallback(BaseCallbackHandler):
    def __init__(self, container, expand_new_thoughts=False):
        self.container = container
        self.expand_new_thoughts = expand_new_thoughts

    def on_llm_new_token(self, token: str, **kwargs):
        self.container.write(token, end="")

    def on_llm_end(self, response, **kwargs):
        self.container.write("\n")

# -------------------------------
# Tools
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

search = DuckDuckGoSearchRun(name="Search")

tools = [search, arxiv, wiki]

# -------------------------------
# Streamlit UI
st.title("🔎 LangChain - Chat with search")
st.markdown("""
Talk with your base LangChain agent and see its responses in real time!  
Build smarter assistants 🤖✨ with your own tools and logic.
""")

# Sidebar API key
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")
if not api_key:
    st.info("Please provide a valid Groq API key to continue")
    st.stop()

# -------------------------------
# Initialize messages
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]

# Display chat history
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

# -------------------------------
# Initialize agent once
if "search_agent" not in st.session_state:
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="meta-llama/llama-4-scout-17b-16e-instruct",
        streaming=True
    )
    st.session_state["search_agent"] = create_agent(llm, tools=tools)

search_agent = st.session_state["search_agent"]

# -------------------------------
# Chat input
if prompt := st.chat_input(placeholder="Ask me anything..."):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Streamlit callback
    with st.chat_message("assistant"):
        st_cb = StreamlitCallback(st.container())
        response = search_agent.invoke({"messages": st.session_state["messages"]}, callbacks=[st_cb])

        # Safely extract output text
        if isinstance(response, dict) and "output_text" in response:
            output_text = response["output_text"]
        else:
            output_text = str(response)

        st.session_state["messages"].append({"role": "assistant", "content": output_text})
        st.write(output_text)

      