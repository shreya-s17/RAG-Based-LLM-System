# from langchain_core.tools import Tool
from langchain_classic.agents import initialize_agent, Tool
from langchain_openai import ChatOpenAI
from backend.app.rag import get_retriever
from langchain_core.prompts import PromptTemplate
from backend.app.rag import get_retriever

llm = ChatOpenAI(temperature=0)

# =========================
# 🔹 TOOL: RAG RETRIEVER
# =========================
def retrieve_context(query):
    retriever = get_retriever()
    docs = retriever.invoke(query)
    return "\n".join([doc.page_content for doc in docs])


# =========================
# 🧠 1. PLANNER AGENT
# =========================
planner_prompt = PromptTemplate.from_template("""
You are a planner agent.

Break the user query into clear step-by-step tasks.

User Query:
{query}

Output ONLY steps in numbered format.
""")

def planner_agent(query):
    prompt = planner_prompt.format(query=query)
    plan = llm.invoke(prompt)
    return plan


# =========================
# ⚙️ 2. EXECUTOR AGENT
# =========================
executor_prompt = PromptTemplate.from_template("""
You are an executor agent.

You have access to:
- Document context
- Tools like Python execution

Steps to follow:
{plan}

Relevant Context:
{context}

Execute the steps and generate a final answer.

Be detailed and accurate.
""")

def executor_agent(query, plan):
    context = retrieve_context(query)

    prompt = executor_prompt.format(
        plan=plan,
        context=context
    )

    result = llm.invoke(prompt)
    return result


# =========================
# 🔍 3. CRITIC AGENT
# =========================
critic_prompt = PromptTemplate.from_template("""
You are a critic agent.

Your job:
- Verify factual correctness
- Ensure answer matches context
- Improve clarity

Original Query:
{query}

Answer:
{answer}

Context:
{context}

If answer is good → return as is.
If not → improve it.
""")

def critic_agent(query, answer):
    context = retrieve_context(query)

    prompt = critic_prompt.format(
        query=query,
        answer=answer,
        context=context
    )

    final_answer = llm.invoke(prompt)
    return final_answer


# =========================
# 🚀 MASTER PIPELINE
# =========================
def run_multi_agent(query):
    print("\n🧠 PLANNING...")
    plan = planner_agent(query)
    print(plan)

    print("\n⚙️ EXECUTING...")
    draft = executor_agent(query, plan)
    print(draft)

    print("\n🔍 CRITIQUING...")
    final = critic_agent(query, draft)

    return {
        "plan": plan,
        "draft": draft,
        "final": final
    }

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
        ),
        # Tool(
        #     name="Planner",
        #     func=run_multi_agent,
        #     description="Breaks down the query into step-by-step tasks"
        # )
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