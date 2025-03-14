import streamlit as st
import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain.agents import AgentType, initialize_agent, load_tools

# Securely Input API Token
st.sidebar.title("Type here")
api_token = "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"

if api_token:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_token

    # Initialize LLM
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct",
        task="text-generation",
        temperature=0.6
    )

    # Load tools
    tools = load_tools(['wikipedia', 'llm-math'], llm=llm)

    # Initialize agent with handle_parsing_errors=True
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True  # This instructs the agent to handle output parsing errors.
    )

    # User input from Sidebar
    user_text = st.sidebar.text_input("Ask a question")

    if user_text:
        res = agent.run(user_text)
        st.write("### Response:")
        st.write(res)
    else:
        st.write("Enter a question in the sidebar.")

else:
    st.warning("Please enter your Hugging Face API Token.")
