  # Load variables from a local .env file into environment (e.g., GOOGLE_API_KEY).
from dotenv import load_dotenv 

# Pydantic is used to define a typed schema for structured outputs.
from pydantic import BaseModel 

# Optional providers you could switch to (not used in this run).
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic 
from langchain_deepseek import ChatDeepSeek

# Active provider in this script: Google Gemini via LangChain.
from langchain_google_genai import ChatGoogleGenerativeAI

# Prompt building utilities for multi-message prompts.
from langchain_core.prompts import ChatPromptTemplate

# Parser that enforces the LLM to return data matching a Pydantic model.
from langchain_core.output_parsers import PydanticOutputParser 

# Agent factory + executor to let the LLM call tools.
from langchain.agents import create_tool_calling_agent, AgentExecutor

# Your custom tools (web search, Wikipedia, save-to-file) defined elsewhere.
from tools import search_tool, wikipedia_tool, save_tool 

# Reads .env and sets environment variables for the process.
load_dotenv()


# Define the shape of the final response your agent should produce.
# The parser below will try to coerce the LLM output into this schema.
class ResearchResponse(BaseModel):
    topic: str          # Short topic name or title
    summary: str        # Plain-English summary of findings
    sources: list[str]  # List of source strings or URLs
    tools: list[str]    # Names of tools the agent claims to have used



# Examples of other LLM backends you might try (currently commented out).
# llm0 = ChatDeepSeek(model="deepseek-chat")
# llm1 = ChatAnthropic(model="claude-3-5-sonnet-20241022")

# Instantiate the active LLM client. Requires GOOGLE_API_KEY in your .env.
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Create a parser that will validate and parse the model's output
# into the ResearchResponse Pydantic schema.
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

# Build a chat-style prompt. It includes:
# - a system message with instructions and "{format_instructions}" placeholder
# - a spot for chat history
# - the human's query
# - a scratchpad for the agent to think and record tool calls
#
# NOTE: The entries ("placeholder", "...") here are literal strings.
# In LangChain, the usual pattern is to use MessagesPlaceholder objects,
# but you asked not to modify code; this is just a comment for awareness.
prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
        """
        You are a research assistant that will help generate a research paper.
        Answer the user query and use neccsessary tools.
        Wrap the output in this format and provide no other text\n{format_instructions}
        """,
        ),
        ("placeholder", "{chat_history}"),
        ("human","{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

# Register the tools that the agent is allowed to call.
tools = [search_tool, wikipedia_tool, save_tool]

# Create an agent that knows how to call tools while following the prompt.
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
    
)

# Wrap the agent with an executor that handles the full call lifecycle,
# including tool routing and returning the final "output" string.
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
)

# Basic CLI input to capture the user's research question.
query = input("Enter your research query: ")
# response = llm.invoke("Who is the one who creates the earth?")

# Execute the agent with the provided "query".
# NOTE: The key here is "query" because your prompt expects {query}.
raw_response = agent_executor.invoke({"query": query})

# print(raw_response)

# Try to parse the agent's final output into the ResearchResponse schema.
# IMPORTANT: In many LangChain executors, raw_response["output"] is a STRING.
# Indexing it like [0]["text"] will often raise a TypeError.
try:
    structured_response = parser.parse(raw_response.get("output")[0]["text"])
    print("Structured Response:", structured_response)
except Exception as e: 
    print("Error parsing response:", e, "Raw response:", raw_response)
