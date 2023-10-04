import os
from apikey import apikey
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

os.environ["OPENAI_API_KEY"] = apikey
st.title("Chat With A Document")
loader = TextLoader("./constitution.txt")
documents = loader.load()
print(documents)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

#to see chunks
#st.write(chunks[0])
#st.write(chunks[1])

embeddings = OpenAIEmbeddings()
vector_store = Chroma.from_documents(chunks, embeddings)

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
retriever = vector_store.as_retriever()
chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

question = st.text_input("Ask a question")

if question:
    response = chain.run(question)
    st.write(response)