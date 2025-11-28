from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
load_dotenv()

model=ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite')

chat_history=[
    SystemMessage(content="You are a helful assistant.")
]

while True:
    user_input=input("You: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input.lower() == 'exit':
        break
    response=model.invoke(chat_history)
    chat_history.append(AIMessage(content=response.content))
    print("AI:", response)
print(response)