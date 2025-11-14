from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder


#chat_template
chat_template=ChatPromptTemplate.from_messages([
    ("system", "You are a helful customer support agent."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")

])

chat_history=[]
# load file
with open("chat_history.txt", "r") as f:
    chat_history.extend(f.readlines())

print(chat_history)

response=chat_template.invoke({
    'chat_history': chat_history,
    "input": "Where is my refund?"
})

print(response)