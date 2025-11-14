from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()

model=ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite')

templete= ChatPromptTemplate.from_messages([
    ("system", "You are a helful assistant."),
    ("human", "{input}")
        # SystemMessage(content="You are a helful {domain} assistant."),
        # HumanMessage(content="explain me {topic}")
])


response=model.invoke(templete.invoke({
    "input":"explain me chatbot"
}))
print(response)

