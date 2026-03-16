
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import RetrievalQA
import tempfile


load_dotenv(override=True)

google_api_key=os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title="RAG Document Chat",layout="wide")

st.title("Text only Pdf Assisstant")
st.write("Upload the document")

upload_file=st.sidebar.file_uploader("Upload the file",type="pdf")

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLm-L6-v2")

def load_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature=0.1,google_api_key=google_api_key)

llm=load_llm()
embeddings=load_embeddings()

if upload_file:
    with tempfile.TemporaryDirectory() as temp_file:
        temp_pdf_path=os.path.join(temp_file,"input.pdf")
        with open(temp_pdf_path,"wb") as f:
            f.write(upload_file.getvalue())

        loader=PyPDFLoader(temp_pdf_path)
        docs=loader.load()

        text_splitter=CharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        chunks=text_splitter.split_documents(docs)

        # @st.cache_resource
        def vector_database(_doc,_embeddings):
            return FAISS.from_documents(_doc,_embeddings)
        
        vector_store=vector_database(chunks,embeddings)
        st.success("PDF Proceeded ! U can ask the questions")

    query=st.text_input("Ask your query")

    if query is not None and st.button("Search"):
        with st.spinner("Searching...."):
           qa_chain=RetrievalQA.from_chain_type(
               llm=llm,
               chain_type="stuff",
               retriever=vector_store.as_retriever()
           )
           response=qa_chain.invoke(query)
           st.write(response["result"])
else:
    st.info("Please Provide the pdf")
