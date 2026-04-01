# from langchain_core.tools import Tool
from langchain_classic.agents import initialize_agent, Tool
from langchain_openai import ChatOpenAI
from backend.app.rag import get_retriever

def rag_tool_func(query):
    retriever = get_retriever()
    docs = retriever.invoke(query)
    return "\n".join([doc.page_content for doc in docs])

def python_tool(code):
    try:
        local_vars = {}
        exec(code, {}, local_vars)
        return str(local_vars)
    except Exception as e:
        return str(e)

def build_agent():
    llm = ChatOpenAI(temperature=0)

    tools = [
        Tool(
            name="Document Search",
            func=rag_tool_func,
            description="Search information from uploaded documents"
        ),
        Tool(
            name="Python Executor",
            func=python_tool,
            description="Executes Python code"
        )
    ]

    agent = initialize_agent(
        tools,
        llm,
        agent="zero-shot-react-description",
        verbose=True
    )

    # agent = createAgent({
    #     model: "gpt-4.1",
    #     tools: tools,
    # })
    
    return agent