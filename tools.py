# Tools and utilities from the LangChain community packages.
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
# Base Tool wrapper used to expose Python callables to an agent.
from langchain.tools import Tool
# For timestamping saved output.
from datetime import datetime, timezone

def save_to_txt(data: str, filename: str = "research_output.txt") -> str:
    """
    Append text to a local file with a timestamp banner.
    Returns a short success message with the target filename.
    """
    timestamp = datetime.now().astimezone(timezone.utc).isoformat(timespec="seconds")
    formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"

    # Open the file in append mode and write the formatted text.
    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_text)
        
    return f"Data successfully saved to {filename}"

# Expose the save function as a LangChain Tool that an agent can call.
save_tool = Tool(
    name="save_text_to_file",
    func=save_to_txt,
    # NOTE: This description currently mentions "Search the web",
    # but the tool actually SAVES text to a file. Description and behavior do not match.
    description="Search the web for relevant information",
)

# Instantiate a DuckDuckGo search runner (community tool).
search = DuckDuckGoSearchRun()

# Expose a search tool. 
# NOTE: This tool is wired to save_to_txt (not to the 'search' object above).
# That means calling this tool will append text to a file instead of performing a web search.
search_tool = Tool(
    name="search_text_to_file",
    func=save_to_txt,
    # Description says it is useful for current events, which matches a search tool,
    # but again the function points to save_to_txt. Name/description vs. behavior mismatch.
    description="useful for when you need to answer questions about current events or the current state",
)

# Configure a lightweight Wikipedia API wrapper (limit results and content length).
api_wrapper = WikipediaAPIWrapper(top_k_results=3,doc_content_chars_max=2000)

# Create the Wikipedia query tool using the configured wrapper.
wikipedia_tool = WikipediaQueryRun(api_wrapper=api_wrapper)
