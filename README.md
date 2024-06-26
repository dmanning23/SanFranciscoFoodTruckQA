# San Francisco Food Truck Q&A
Have your questions answered about food trucks in San Francisco

Try it out:
https://sanfranciscofoodtruckapp-cgqjbjws9g4lgpssbqgej3.streamlit.app/

This app loads a document about San Francisco food trucks, and allows the user to ask questions about the data. It uses the OpenAI API and several experimental features of the LangChain project to find relevant data points and answer the questions. 

This project can answer simple questions, but this technology really shines when loaded with a more robust data set. For example, if loaded with complicated financial data, it would allow the user to slice and dice the data to find insights and even create graphs in a fraction of the time of an entire team of financial analysts or data scientists.

Given more time, I would probably do something like add additional data sources and agents to the AI chain. For example, if there was geographic database that could be attached to a Google Maps agent, the chat bot could combine that with the address data to answer location based queries like "Find taco trucks near me".

![AppRunning](./Screenshot.png?raw=true "AppRunning")
