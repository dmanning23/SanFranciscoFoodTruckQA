# ChatWithDocument
Chat with a document using Python, LangChain, ChromaDB, and LLM

First install the necessary python libraries:
```
pip3 install chromadb streamlit langchain openai tiktoken
```

Set your OpenAI API key in the apikey.py file
You'll need to have credits available in your OpenAI account to run this program, or you will hit the 429 rate limit error.
It costs about a penny to load the document, parse through the OpenAI model, and store in ChromaDB. (YMMV)

Use streamlit to run the program:
```
streamlit run app.py
```
