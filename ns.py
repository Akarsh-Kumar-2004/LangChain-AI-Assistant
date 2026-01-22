import streamlit as st
import requests
import os
from langchain_community.llms import HuggingFaceEndpoint
from huggingface_hub import InferenceClient

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

llm = InferenceClient(
    model="HuggingFaceH4/zephyr-7b-beta",
    token="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"
)

def fetch_text_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text(separator=' ')
        text = " ".join(text.split()[:10000])

        return text if text else "No readable content found."
    except Exception as e:
        return f"Error fetching data from {url}: {str(e)}"



def generate_answer(news, question):
    prompt = f"Based on the following news article:\n{news}\n\nAnswer the question: {question}"
    response = llm.text_generation(prompt, max_new_tokens=256)
    return response.strip()

st.set_page_config(page_title="News Analysis")
st.title("Article Question Answering Chatbot ")


st.sidebar.header("Enter News URLs & Question")
urls = st.sidebar.text_area("Enter News URLs (One per line)", placeholder="https://example.com/news1\nhttps://example.com/news2")
question = st.sidebar.text_input("Enter your question")

if st.sidebar.button("Get Answer"):
    with st.spinner("Fetching news and processing..."):
        news_texts = []
        url_list = urls.split("\n")

        for url in url_list:
            url = url.strip()
            if url:
                extracted_text = fetch_text_from_url(url)
                news_texts.append(extracted_text)

        if news_texts:
            combined_news_text = " ".join(news_texts)
            answer = generate_answer(combined_news_text, question)
            
            st.subheader("Extracted News Content")
            for i, text in enumerate(news_texts, 1):
                st.markdown(f"**News {i}:**")
                st.write(text[:500] + "...")  
                
            st.subheader(" AI-Generated ")
            st.write(answer)
        else:
            st.error("No valid news content found. Please check the URLs.")


