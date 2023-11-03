# ChatWithConstitution
Chat with a document using Python, LangChain, QDrant, and LLM

This app loads a document, stores the embeddings in a vector datastore (QDrant in this example), and allows the user to chat with the document and ask it questions. The user's chat history is stored in the session, so it can be reset by using the Clear button or refreshing the browser. The app will display the relevent sections it found in the source text, with relevency scores.


