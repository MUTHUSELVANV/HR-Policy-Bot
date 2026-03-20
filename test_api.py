from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
response = llm.invoke("Say exactly this: Day 1 is working!")
print(response.content)