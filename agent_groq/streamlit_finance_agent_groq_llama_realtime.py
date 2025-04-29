import streamlit as st
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools
from dotenv import load_dotenv

load_dotenv()

# Create the agents
def create_agents():
    web_agent = Agent(
        name="Web Agent",
        model=Groq(id="meta-llama/llama-4-maverick-17b-128e-instruct"),
        tools=[DuckDuckGo()],
        instructions=["Always include sources"],
        show_tool_calls=True,
        markdown=True
    )

    finance_agent = Agent(
        name="Finance Agent",
        role="Get financial data",
        model=Groq(id="meta-llama/llama-4-maverick-17b-128e-instruct"),
        tools=[YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            company_info=True
        )],
        instructions=["Use tables to display data"],
        show_tool_calls=True,
        markdown=True
    )

    agent_team = Agent(
        model=Groq(id="meta-llama/llama-4-maverick-17b-128e-instruct"),
        team=[web_agent, finance_agent],
        instructions=[
            "Always include sources",
            "Use tables to display data"
        ],
        show_tool_calls=True,
        markdown=True
    )

    return agent_team

# Streamlit UI
st.title("🧠 Finance Agent using Groq + Phi")
query = st.text_input("Enter your financial question:", value="Get the latest financial data for HDFCBANK including market cap, P/E ratio, EPS, dividend yield, current price, and return on equity in Indian")

if st.button("Run Agent"):
    agent_team = create_agents()

    with st.spinner("Running agents and fetching data..."):
        conversation = agent_team.run(query)
        # Only show last assistant message
        assistant_messages = [msg for msg in conversation.messages if msg.role == "assistant" and msg.content and msg.content.strip().lower() != "none"]
        if assistant_messages:
            st.markdown("**🤖 Assistant (Final Summary):**")
            st.markdown(assistant_messages[-1].content, unsafe_allow_html=True)

