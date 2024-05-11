import os
import streamlit as st
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import pandas as pd
from langchain_openai import OpenAI

def main():
    os.environ["OPENAI_API_KEY"] = st.secrets["openAIapikey"]
    st.set_page_config(
        page_title="San Francisco Food Trucks Q&A",
        page_icon="🌮")
    
    st.text("Ask any question about food trucks in San Francisco!")
    st.text("Examples:")
    st.text('"Which food trucks serve tacos?"')
    st.text('"What food trucks serve vegan food?"')
    st.text('"Can you tell me something interesting about this data?"')

    #Read the csv file into a Pandas dataframe
    df = pd.read_csv("./Mobile_Food_Facility_Permit.csv")

    agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True)

    container = st.container()
    with container:
        with st.form(key="my form", clear_on_submit=True):
            user_input  = st.text_area(label="Question: ", key="input", height = 100)
            submit_button = st.form_submit_button(label="Ask")

        if submit_button and user_input:

            with st.chat_message('user'):
                st.markdown(user_input)

            with st.spinner("Thinking..."):
                response = agent.invoke(user_input)
            
                with st.chat_message('assistant'):
                    st.markdown(response["output"])

if __name__ == "__main__":
    main()