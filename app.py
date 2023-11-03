import os
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)

@st.cache_resource
def InitializeDocument():
     #Load the document
    loader = TextLoader("./constitution.txt")
    documents = loader.load()

    #Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    #Get the embeddings for the list of chunks and store in the vectordb
    embeddings = OpenAIEmbeddings()
    return Qdrant.from_documents(
        chunks,
        embeddings,
        location=":memory:",  # Local mode with in-memory storage only
        collection_name="my_documents",)

def main():
    #os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    st.set_page_config(
        page_title="Chat With A Document",
        page_icon="😺")
    
    #setup the sidebar
    st.sidebar.title("Options")

    #Create the memory object
    if "memory" not in st.session_state:
        st.session_state["memory"]=ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    memory=st.session_state["memory"]

    #add a button to the sidebar to start a new conversation
    clear_button = st.sidebar.button("New Conversation", key="clear")
    if (clear_button):
        print("Clearing memory")
        memory.clear()

    vector_store = InitializeDocument()

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
    retriever=vector_store.as_retriever()
    crc = ConversationalRetrievalChain.from_llm(llm,
                                                retriever,
                                                memory=memory)

    container = st.container()
    with container:
        with st.form(key="my form", clear_on_submit=True):
            user_input  = st.text_area(label="Question: ", key="input", height = 100)
            submit_button = st.form_submit_button(label="Ask")

        if submit_button and user_input:

            #Use the embedding function to get the similar documents
            documents = vector_store.similarity_search_with_score(user_input)
            st.write(f"There were {len(documents)} matching documents found:")
            for document, score in documents:
                st.write(document)
                st.subheader(f"Score: {score}")

            with st.spinner("Thinking..."):
                question = {'question': user_input}
                response = crc.run(question)
            
            #write the ressponse
            st.subheader(response)

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
    
if __name__ == "__main__":
    main()