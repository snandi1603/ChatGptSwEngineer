from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime

search = DuckDuckGoSearchRun()
search_tool = Tool(
    name = "search",
    func = search.run,
    description="Search the web for information"
)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100, wiki_client=None)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

def save_to_text_file(content: str, filename: str = "Output/research_output_def.txt"):
    time_stamp = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    formatted_text = f"---- Research Output ----\n Timestamp: {time_stamp}\n\n{content}\n"
    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_text)
    return f"Data Successfully written to {filename}"

save_tool = Tool(
    name="save_to_text_file",
    func=save_to_text_file,
    description="Save research output to a text file"
)