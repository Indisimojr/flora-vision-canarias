import base64
from n3_tools import search_plant_info

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

tools = [search_plant_info]

llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    model="ivands/qwen3-vl-plant-tuned" # qwen/qwen3-vl-4b
)

llm = llm.bind_tools(tools)
tools_by_name = {
    tool.name: tool 
    for tool in tools
}

system_prompt = """
You're an assistant specialized in endemic vegetation from the Canary Islands. Always answer in the same language as the user.

You have access to a tool called `search_plant_info` that searches a local corpus about Canary Islands' plants.
Use it when the user asks for factual information about a plant, species, morphology, habitat, distribution, flowers, leaves, fruits, uses or general details.
- If the user only asks for identification, don't call it.
- Only use it when you know the species name.

If you don't get any content from the corpus, mention it, but at least describe the visual features of the plant.
"""

def image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def build_message_content(text=None, files=None):
    content = []

    if files:
        for file in files:
            image_b64 = image_to_base64(file)

            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_b64}"
                }
            })

    if text:
        content.append({
            "type": "text",
            "text": text
        })

    return content

def chat(message, history, temperature, max_tokens):
    messages = [SystemMessage(content=system_prompt.strip())]

    # Añadimos los mensajes en el historial a "messages".
    for history_message in history:
        role = history_message["role"]
        content = history_message["content"]

        if role == "user":
            messages.append(HumanMessage(content=content))

        elif role == "assistant":
            messages.append(AIMessage(content=content))

    # Añadimos el mensaje nuevo, del usuario, a la lista.
    current_content = build_message_content(
        message.get("text", ""),
        message.get("files", [])
    )

    messages.append(
        HumanMessage(content=current_content)
    )

    # Y hacemos 2 llamadas: una para ver si necesita información del RAG y otra para utilizar la información, si la recibe.
    first_response = llm.invoke(
        messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    messages.append(first_response)

    if first_response.tool_calls:
        for tool_call in first_response.tool_calls:
            selected_tool = tools_by_name[tool_call["name"]]

            tool_result = selected_tool.invoke(tool_call["args"])

            messages.append(
                ToolMessage(
                    content=tool_result,
                    tool_call_id=tool_call["id"]
                )
            )

        final_response = llm.invoke(
            messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

        return final_response.content

    return first_response.content
