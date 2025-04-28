# encoding: utf-8
"""
@author:  raufschlaeger
"""

from ollama import chat, ChatResponse


def fix_graph(graph: str):
    messages = [{
        'role': 'user',
        'content': (
            "Fix the JSON graph. "
            "Example format: {\"nodes\":[{\"id\":\"person\",\"attributes\":[...]}], "
            "\"edges\":[{\"source\":\"...\",\"target\":\"...\",\"relation\":\"...\"}]} "
            "Requirements:\n"
            "- keep attributes\n"
            "- remove redundancies"
            "- Nodes: id + attributes list\n"
            "- Edges: source/target/relation\n"
            "- Only use one word per node/source/target/attribute\n"
             "- Output only the valid revised JSON, and no explanation or notes\n"
            f"Text: {graph}"
        ),
    }]

    response: ChatResponse = chat(model="phi4", messages=messages)
    return response.message.content
