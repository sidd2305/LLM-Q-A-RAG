import os
import pickle
import time
from langchain.llms import HuggingFaceEndpoint
import pandas as pd
from langchain_community.embeddings import FakeEmbeddings
from io import StringIO
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from newspaper import Article
from langchain.schema import Document
from langchain.llms import OpenLLM 
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from pypdf import PdfReader  # Use pypdf for handling PDF files



load_dotenv()  # Load environment variables from .env

st.title("Siddhanth`s RAG Tool 📈")

# Sidebar for selecting between URL processing and file upload
st.sidebar.title("Options")
app_mode = st.sidebar.radio("Choose an option", ["Process URLs", "Upload File"])

# URL Processing
if app_mode == "Process URLs":
    st.write("Enter a few websites(maybe news/articles) on the left panel that youd like to use as reference for the LLM Q&A System!")
    urls = [st.sidebar.text_input(f"URL {i+1}") for i in range(3)]
    process_url_clicked = st.sidebar.button("Process URLs")
    file_path = "faiss_store_openai.pkl"

    main_placeholder = st.empty()
    llm = HuggingFaceEndpoint(
    endpoint_url="https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    huggingfacehub_api_token="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",
    max_new_tokens=7000
)

    # llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.9, max_tokens=500)
    # llm = OpenLLM(model="wizardlm-13B", temperature=0.9, max_tokens=500)

    if process_url_clicked:
        data = []
        main_placeholder.text("Data Loading...Started...✅✅✅")

        for url in urls:
            if url:
                try:
                    article = Article(url)
                    article.download()
                    article.parse()
                    data.append(Document(page_content=article.text, metadata={"source": url}))
                except Exception as e:
                    st.write(f"Error fetching or processing {url}, exception: {e}")

        if data:
            text_splitter = RecursiveCharacterTextSplitter(
                separators=['\n\n', '\n', '.', ','],
                chunk_size=2500
            )
            main_placeholder.text("Text Splitter...Started...✅✅✅")
            docs = text_splitter.split_documents(data)

            # Replace OpenAIEmbeddings with HuggingFaceEmbeddings
            embeddings = FakeEmbeddings(size = 500) 

            vectorstore_openai = FAISS.from_documents(docs, embeddings)
            main_placeholder.text("Embedding Vector Started Building...✅✅✅")
            time.sleep(2)

            with open(file_path, "wb") as f:
                pickle.dump(vectorstore_openai, f)

    query = st.text_input("Question: ")
    if query:
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                vectorstore = pickle.load(f)
                chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
                result = chain({"question": query}, return_only_outputs=True)

                st.header("Answer")
                st.write(result["answer"])

                sources = result.get("sources", "")
                if sources:
                    st.subheader("Sources:")
                    sources_list = sources.split("\n")
                    for source in sources_list:
                        st.write(source)

# File Upload
elif app_mode == "Upload File":
    st.write("Upload a file to use as a reference for the LLM Q&A System")
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "txt", "pdf"])

    if uploaded_file:
        file_path = "faiss_store_openai.pkl"
        llm = HuggingFaceEndpoint(
    endpoint_url="https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    huggingfacehub_api_token="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",
    max_new_tokens=8000
)

        # llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.9, max_tokens=500)
        # llm = OpenLLM(model="wizardlm-13B", temperature=0.9, max_tokens=500)
        main_placeholder = st.empty()
        if uploaded_file.type == "text/csv":
            df = pd.read_csv(uploaded_file)
            data = [Document(page_content=row.to_string(), metadata={"source": "uploaded_csv"}) for _, row in df.iterrows()]
        elif uploaded_file.type == "text/plain":
            data = [Document(page_content=uploaded_file.read().decode("utf-8"), metadata={"source": "uploaded_txt"})]
        elif uploaded_file.type == "application/pdf":
            reader = PdfReader(uploaded_file)
            data = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    data.append(Document(page_content=text, metadata={"source": "uploaded_pdf"}))

        if data:
            text_splitter = RecursiveCharacterTextSplitter(
                separators=['\n\n', '\n', '.', ','],
                chunk_size=1000
            )
            main_placeholder.text("Text Splitter...Started...✅✅✅")
            docs = text_splitter.split_documents(data)

            # Replace OpenAIEmbeddings with HuggingFaceEmbeddings
            embeddings = FakeEmbeddings(size = 500) 

            vectorstore_openai = FAISS.from_documents(docs, embeddings)
            main_placeholder.text("Embedding Vector Started Building...✅✅✅")
            time.sleep(2)

            with open(file_path, "wb") as f:
                pickle.dump(vectorstore_openai, f)

        query = st.text_input("Question: ")
        if query:
            if os.path.exists(file_path):
                with open(file_path, "rb") as f:
                    vectorstore = pickle.load(f)
                    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
                    result = chain({"question": query}, return_only_outputs=True)

                    st.header("Answer")
                    st.write(result["answer"])

                    sources = result.get("sources", "")
                    if sources:
                        st.subheader("Sources:")
                        sources_list = sources.split("\n")
                        for source in sources_list:
                            st.write(source)
