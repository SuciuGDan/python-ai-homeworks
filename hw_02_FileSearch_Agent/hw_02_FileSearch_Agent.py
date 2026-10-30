import os
import json
import subprocess
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent

# tool 1: lists all the files in the project
def list_files() -> str:
    try:
        result = subprocess.run(
            "dir /s",
            shell=True,
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            timeout=60,
        )
        return result.stdout or "No files found"
    except Exception as e:
        return f"Error listing files: {e}"


# tool 2: searches a keyword inside the content of all files
def search_keyword(keyword: str) -> str:
    try:
        result = subprocess.run(
            ["findstr", "/s", "/i", keyword, "*.*"],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            timeout=60,
        )
        return result.stdout or "No file contains this keyword"
    except Exception as e:
        return f"Error searching keyword: {e}"


def complete(client: OpenAI, messages: list[dict], tools: list[dict]):
    response = client.chat.completions.create(
        messages=messages,
        model=os.getenv("OPEN_ROUTER_MODEL_NAME"),
        tools=tools,
        tool_choice="auto",
    ).choices[0].message
    return response


tools = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "Lists all the files in the project (repository)",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_keyword",
            "description": "Searches a keyword in the content of all files in the project "
                           "and returns the names of the files that contain it",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string", "description": "The keyword to search for"}
                },
                "required": ["keyword"],
            },
        },
    },
]

SYSTEM_PROMPT = (
    "You are an agent that searches in a project. "
    "Always follow these steps in order: "
    "1. call the list_files tool first. "
    "2. then call the search_keyword tool with the keyword 'pytest'. "
    "3. finally, answer with the list of files in which the keyword was found."
)


def run_agent(client: OpenAI, user_input: str, messages = list[dict]) -> str:
    """Runs the agent until it stops calling tools, returns its final text"""

    ###################### for a response with memory ##############:
    if messages is None:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    messages.append({"role": "user", "content": user_input})

    ################# for a response without memory #################3:
    #messages = [
     #   {"role": "system", "content": SYSTEM_PROMPT},
  #      {"role": "user", "content": user_input},
  #  ]

    while True:
        response = complete(client, messages, tools)
        # the assistant message (with its tool_calls) must be in the history
        # before the tool results
        messages.append(response.model_dump(exclude_none=True))

        # no more tools requested -> this is the final answer
        if not response.tool_calls:
            return response.content

        for tool_call in response.tool_calls:
            tool_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments or "{}")

            if tool_name == "list_files":
                output = list_files()
            elif tool_name == "search_keyword":
                output = search_keyword(args.get("keyword", ""))
            else:
                output = f"Error: unknown tool {tool_name}"

            print(f"[tool called] {tool_name}")
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": output,
            })


def main():
    load_dotenv()
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPEN_ROUTER_API_KEY"),
    )
   # answer = run_agent(client, "Find the files that contain the keyword pytest")
   # print("Agent said:", answer)

    # for a chat with history of the messages:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        inp = input("You: ")
        if inp == "quit":
            break
        answer = run_agent(client, inp, messages)
        print("Agent said:", answer)
        print(f"Message count: {len(messages)}")

if __name__ == "__main__":
    main()