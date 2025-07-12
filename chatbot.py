# Importing the necessary libraries
import os
from dotenv import load_dotenv


import streamlit as st
from llama_index.llms.groq import Groq

# Loading the API key from environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Function to generate chatbot response
def chat_qa(prompt):
    ilm = Groq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY, temperature=0.5)
    response = ilm.complete(prompt)
    return response

# Streamlit UI
st.title(f"*My AI :green[Chatbot]* :sparkles:")

# Initializing chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Displaying chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input Section
if prompt := st.chat_input("Ask any question here !"):
    # Displaying user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    #Chat Responses
    response = chat_qa(prompt)

    # Display assistant message
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
