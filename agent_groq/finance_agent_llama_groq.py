from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from dotenv import load_dotenv

load_dotenv()

# Define the function tool
def get_company_symbol(company: str) -> str:
    """Returns the stock symbol for a given company name."""
    symbols = {
        "Infosys": "INFY",
        "Tesla": "TSLA",
        "Apple": "AAPL",
        "Microsoft": "MSFT",
        "Amazon": "AMZN",
        "Google": "GOOGL",
        "Reliance": "RELIANCE",
        "Tata Consultancy Services Ltd": "TCS",
        "HDFC Bank": "HDFCBANK",
        "State Bank of India": "SBIN",
    }
    return symbols.get(company, "Unknown")

# Initialize the agent
agent = Agent(
    model=Groq(id="meta-llama/llama-4-maverick-17b-128e-instruct"),
    tools=[
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            stock_fundamentals=True
        ),
        get_company_symbol  # use function directly
    ],
    instructions=[
        "Use tables to display data.",
        "If you need to find the symbol for a company, use the get_company_symbol tool.",
    ],
    show_tool_calls=True,
    markdown=True,
    #debug_mode=True,
)

# Company names to compare
companies = ["HDFC Bank", "State Bank of India", "Infosys"]
symbols = [get_company_symbol(c) for c in companies]

# Check for unknown symbols
if "Unknown" in symbols:
    raise ValueError(f"One or more companies returned unknown symbols: {symbols}")

# Compose query
query = f"Summarize and compare analyst recommendations and fundamentals for {symbols[0]} {symbols[1]} and {symbols[2]}. Show in tables."

# Run the query
agent.print_response(query, stream=True)
