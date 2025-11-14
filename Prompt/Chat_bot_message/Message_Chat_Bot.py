from langchain_core.messages import HumanMessage, SystemMessage,AIMessage
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()

model=ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite')

Meessage=[
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Hi, how are you?")
]

response=model.invoke(Meessage)

print(response)