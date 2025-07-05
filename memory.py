import os
from dotenv import load_dotenv
load_dotenv()

from llama_index.llms.groq import Groq
from llama_index.core import Settings
from llama_index.memory.mem0 import Mem0Memory  
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import FunctionCallingAgent
from llama_index.core.chat_engine import SimpleChatEngine
import nest_asyncio

# Constants and API keys from environment
MODEL_NAME = "llama-3.3-70b-versatile"
LLM_API_KEY = os.getenv("GROQ_API_KEY")
MEM0_API_KEY = os.getenv("MEM0_API_KEY")

# LLM Setup
def get_llm(model_name, api_key):
    return Groq(model_name, api_key)

# Memory Setup
def get_memory(api_key, search_msg_limit=5):
    context = {"user_id": "test_user_1"}
    return Mem0Memory.from_client(api_key=api_key, context=context, search_msg_limit=search_msg_limit)

# Initialize settings
def initialize_settings():
    Settings.llm = get_llm(MODEL_NAME, LLM_API_KEY)
    Settings.memory = get_memory(MEM0_API_KEY, 4)

initialize_settings()

# Define sample functions
def call_fn(name: str):
    print(f"Calling {name}...")

def email_fn(name: str):
    print(f"Emailing {name}...")

# Tools
call_tool = FunctionTool.from_defaults(fn=call_fn)
email_tool = FunctionTool.from_defaults(fn=email_fn)

# FunctionCallingAgent
agent = FunctionCallingAgent.from_tools(
    [call_tool, email_tool],
    llm=Settings.llm,
    memory=Settings.memory,
    verbose=True
)

# Example chat
response = agent.chat("Hi, My name is Shabna.....")
print(response)

# SimpleChatEngine
memory_from_client = get_memory(MEM0_API_KEY, 4)
simple_agent = SimpleChatEngine.from_defaults(llm=Settings.llm, memory=memory_from_client)
