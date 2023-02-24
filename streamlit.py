import streamlit as st
from streamlit_chat import message
import requests
import os
import openai

os.environ["OPENAI_API_KEY"] = str(st.secrets["OPENAI_API_KEY"])

def get_openai_response(input:str):
    data = data + f"\n{input}"
    response = openai.Completion.create(
                model="text-davinci-003",
                prompt=data,
                temperature=0.3,
                max_tokens=200,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
    return response["choices"][0]["text"]

with open("toolsqa.txt") as f:
    data = f.read()   

st.set_page_config(
    page_title="Streamlit Chat - ToolsQA",
    page_icon=":robot:"
)

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

st.header("Streamlit Chat - ToolsQA")
st.markdown("[Github](https://github.com/abhijeetnazar/streamlit-chatbot-demo)")
st.markdown ("This chatbot is used to answer questions for **ToolsQA.com** and available courses and modules.")

def get_text():
    input_text = st.text_input("You: ","Hello, how are you?", key="input")
    return input_text 


user_input = get_text()

if user_input:
    output = get_openai_response(user_input)
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
