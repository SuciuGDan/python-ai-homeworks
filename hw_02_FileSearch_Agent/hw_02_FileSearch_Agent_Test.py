import os
from dotenv import load_dotenv
from openai import OpenAI

from hw_02_FileSearch_Agent import search_keyword, list_files, run_agent


# both files below contain the word "pytest", so we expect to find them
def test_list_files_tool():
    output = list_files()
    assert "hw_02_FileSearch_Agent.py" in output


def test_search_tool():
    # runs only the tool, without the LLM
    output = search_keyword("pytest")
    if "hw_02_FileSearch_Agent.py" in output:
        assert True
    else:
        assert False, f"hw_02_FileSearch_Agent.py not found in: {output}"


def test_agent():
    # runs the whole agent (needs the API key in .env)
    load_dotenv()
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPEN_ROUTER_API_KEY"),
    )
    answer = run_agent(client, "Find the files that contain the keyword pytest")
    print(answer)
    assert "hw_02_FileSearch_Agent.py" in answer