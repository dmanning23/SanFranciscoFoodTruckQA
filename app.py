import os
from apikey import apikey
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)

@st.cache_resource
def InitializeMemory():
    print("resetting memory")
    return ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def SelectModel():
    model = st.sidebar.radio("Choose a model:", ("GPT-3.5", "GPT-4"))
    if model == "GPT-3.5":
        model_name = "gpt-3.5-turbo"
    else:
        model_name = "gpt-4"

    temperature = st.sidebar.slider("Temperature:", 
                                    min_value=0.0, 
                                    max_value=2.0,
                                    value=0.0,
                                    step=0.01)
    
    return ChatOpenAI(model=model_name, temperature=temperature)

def InitializeModel(memory):
    loader = TextLoader("./constitution.txt")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(chunks, embeddings)

    llm = SelectModel()
    retriever=vector_store.as_retriever()
    return ConversationalRetrievalChain.from_llm(llm,
                                                retriever,
                                                memory=memory)

def Run(memory, crc):
    container = st.container()
    with container:
        with st.form(key="my form", clear_on_submit=True):
            user_input  = st.text_area(label="Question: ", key="input", height = 100)
            submit_button = st.form_submit_button(label="Ask")

        if submit_button and user_input:

            with st.spinner("Thinking..."):
                question = {'question': user_input}
                response = crc.run(question)
            
            #write the ressponse
            st.write(response)

            #write the chat history
            variables = memory.load_memory_variables({})
            messages = variables['chat_history']
            for message in messages:
                if isinstance(message, AIMessage):
                    with st.chat_message('assistant'):
                        st.markdown(message.content)
                elif isinstance(message, HumanMessage):
                    with st.chat_message('user'):
                        st.markdown(message.content)
                else:
                    st.write(f"System message: {message.content}")

def main():
    os.environ["OPENAI_API_KEY"] = apikey
    st.set_page_config(
        page_title="Chat With A Document",
        page_icon="😺")
    
    #setup the sidebar
    st.sidebar.title("Options")

    memory = InitializeMemory()

    #add a button to the sidebar to start a new conversation
    clear_button = st.sidebar.button("New Conversation", key="clear")
    if (clear_button):
        print("Clearing memory")
        memory.clear()

    crc = InitializeModel(memory)
    Run(memory, crc)
    
if __name__ == "__main__":
    main()