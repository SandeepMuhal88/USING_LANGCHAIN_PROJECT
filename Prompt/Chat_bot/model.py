import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

model=ChatGoogleGenerativeAI(model='gemini-2.5-flash')

st.header('DEEP AI FOR Research')

user_input = st.text_input('Enter your Prompt')

if st.button('Submit'):
    result = model.invoke(user_input)
    st.write(result.content)