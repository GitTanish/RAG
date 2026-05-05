from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv # environment variables 
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq( model="openai/gpt-oss-120b")

prompt = PromptTemplate(
    template="You're an Anime Nerd, you'll explain {topic} to a fellow nerd.",
    input_variables=["topic"],
)

parser = StrOutputParser()

document_path = Path(__file__).resolve().parents[1] / 'docs' / 'Neon_Genesis_Evangelion.txt'
loader = TextLoader(str(document_path), encoding='utf-8')


docs= loader.load()

chain = prompt | llm | parser
print(chain.invoke({"topic":docs[0].page_content[:500]}))