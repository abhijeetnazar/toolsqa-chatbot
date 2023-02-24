import streamlit as st
from streamlit_chat import message
import requests
import os
import faiss
from langchain import OpenAI
from langchain.chains import VectorDBQAWithSourcesChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
import pickle
import argparse

os.environ["OPENAI_API_KEY"] = str(st.secrets["OPENAI_API_KEY"])

with open("vectorstore.pkl", "rb") as f:
    store = pickle.load(f)
index = faiss.read_index("vectorstore.index")
store.index = index
chain = load_qa_with_sources_chain(OpenAI(temperature=0))


st.set_page_config(
    page_title="Streamlit Chat - ToolsQA",
    page_icon=":robot:"
)

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []


def get_response(question:str):
  result = chain({"input_documents": store.similarity_search(question, k=4),"question": question,},return_only_outputs=True,)["output_text"]
  result = result.replace("SOURCES: toolsqa.txt","")
  return result

st.header("Streamlit Chat - ToolsQA")
st.markdown("[Github](https://github.com/abhijeetnazar/streamlit-chatbot-demo)")
st.markdown ("This chatbot is used to answer questions for **ToolsQA.com** and available courses and modules.")

def get_text():
    input_text = st.text_input("You: ","Hello, how are you?", key="input")
    return input_text 


user_input = get_text()

if user_input:
    output = get_response(user_input)
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
