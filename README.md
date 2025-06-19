# 🧠 LangGraph Tutorial: LLM-Based Workflows with Groq + Tools + Memory

This tutorial walks you through building conversational workflows using [LangGraph](https://docs.langchain.com/langgraph/), [LangChain](https://python.langchain.com), and the **Groq** LLM interface. You will learn to create:

- Simple LLM agents,
- Tool-augmented agents,
- Conditional routing logic,
- Stateful conversation memory with `MemorySaver`.

---

## 📦 1. Environment Setup

```python
from dotenv import load_dotenv
import os

# Load environment variables from a `.env` file
load_dotenv()

# Set Groq API Key
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
```

---

## 🤖 2. Load the Groq LLM

```python
from langchain_groq import ChatGroq

# Use DeepSeek's distilled LLaMA model via Groq
model = "deepseek-r1-distill-llama-70b"
llm = ChatGroq(model_name=model)

# Quick test
print(llm.invoke("hi").content)
```

---

## ⚙️ 3. Define a Model Wrapper Function

```python
from langchain_core.messages import BaseMessage
from langgraph.graph import MessagesState

# Wrapper function to invoke the LLM
def call_model(state: MessagesState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {"messages": [response]}
```

---

## ♻️ 4. Simple Workflow Without Tools

```python
from langgraph.graph import StateGraph, START, END

workflow = StateGraph(MessagesState)
workflow.add_node("mybot", call_model)
workflow.add_edge(START, "mybot")
workflow.add_edge("mybot", END)

app = workflow.compile()

# Test the app
input_data = {"messages": ["hi hello how are you?"]}
print(app.invoke(input_data))
```

---

## 📊 5. Visualize the Graph

```python
from IPython.display import Image, display
display(Image(app.get_graph().draw_mermaid_png()))
```

---

## 🔧 6. Adding a Custom Tool

```python
from langchain_core.tools import tool

@tool
def search(query: str) -> str:
    """Custom weather search tool"""
    if "delhi" in query.lower():
        return "The temp is 45 degree and sunny"
    return "The temp is 25 degree and cloudy"
```

---

## 🧠 7. Binding Tools to the LLM

```python
tools = [search]
llm_with_tool = llm.bind_tools(tools)

# Invoke with tool support
response = llm_with_tool.invoke("What is the weather in Delhi?")
print(response.content)
print(response.tool_calls)
```

---

## 🔀 8. Routing with Conditional Tool Calling

```python
def router_function(state: MessagesState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END
```

---

## 🔸 9. Create a Workflow with Conditional Edges and ToolNode

```python
from langgraph.prebuilt import ToolNode

tool_node = ToolNode(tools)
workflow2 = StateGraph(MessagesState)

workflow2.add_node("llmwithtool", call_model)
workflow2.add_node("mytools", tool_node)

workflow2.add_edge(START, "llmwithtool")
workflow2.add_conditional_edges("llmwithtool", router_function, {"tools": "mytools", END: END})

app2 = workflow2.compile()
```

---

## 📈 10. Visualize and Test Tool-Augmented Workflow

```python
display(Image(app2.get_graph().draw_mermaid_png()))
response = app2.invoke({"messages": ["What is the weather in Delhi?"]})
print(response["messages"][-1].content)
```

---

## ♻️ 11. Loop Back Tool to LLM for Multi-Hop

```python
workflow2.add_edge("mytools", "llmwithtool")
app3 = workflow2.compile()
```

```python
for output in app3.stream({"messages": ["What is the weather in New Delhi?"]}):
    for key, value in output.items():
        print(f"Output from {key}:\n{value}\n")
```

---

## 💾 12. Add Memory with `MemorySaver`

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
workflow3 = StateGraph(MessagesState)

workflow3.add_node("llmwithtool", call_model)
workflow3.add_node("mytools", tool_node)
workflow3.add_edge(START, "llmwithtool")
workflow3.add_conditional_edges("llmwithtool", router_function, {"tools": "mytools", END: END})
workflow3.add_edge("mytools", "llmwithtool")

app4 = workflow3.compile(checkpointer=memory)
display(Image(app4.get_graph().draw_mermaid_png()))
```

---

## 🧪 13. Stateful Testing with Threaded Config

```python
config = {"configurable": {"thread_id": "1"}}

# Stream events
events = app4.stream({"messages": ["What is the weather in Indore?"]}, config=config, stream_mode="values")

for event in events:
    event["messages"][-1].pretty_print()
```

---

## 🧠 14. Test Memory Recall

```python
# Continue the conversation
events = app4.stream({"messages": ["In which city the temp was 25 degree?"]}, config=config, stream_mode="values")
for event in events:
    event["messages"][-1].pretty_print()

# Inspect saved memory
memory.get(config)
```

---

## ✅ Conclusion

You've built a powerful **agentic workflow** with:

- 🌐 Groq LLM + DeepSeek model
- 💪 Custom tools
- 🔀 Conditional routing
- ♻️ Memory for long-term context

This pipeline is modular, visualizable, and highly adaptable for research, chatbots, or task-specific agents.

