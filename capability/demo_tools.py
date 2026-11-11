from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool

import json
import math
import os

# ----------------------------------------
from datetime import datetime
from zoneinfo import ZoneInfo

@tool # mandatory decorator: Registers the function as a tool the agent can offer to the model.
def current_time(timezone: str = "Asia/Kolkata") -> str: # timezone: str = "Asia/Kolkata" Becomes the input schema: one text argument, optional, with a default.
    # a clear docstring (so the model knows *when* to use it)(with an `Args:` section) becomes the description. 
    # The model reads it to decide whether this tool fits the question.
    """Get the current date and time in a given IANA timezone. 

    Args: # type hints (so the model knows what arguments to send)
        timezone: IANA timezone name, for example "Asia/Kolkata" or "America/New_York". 
    """
    return datetime.now(ZoneInfo(timezone)).strftime("%A, %d %B %Y %I:%M %p %Z")

# ----------------------------------------

# A simple tool the agent can call
@tool # mandatory decorator: Registers the function as a tool the agent can offer to the model.
def greet(name: str) -> str:
    # a docstring (so the model knows *when* to use it)
    """Greet someone by name."""
    return f"Hello, {name}! Welcome to the world of AI agents."

# ----------------------------------------

import requests

'''
Inside, the function makes an HTTP GET request to the URL with a 5-second timeout. If the server answers, it returns the status code (200 means success). If anything fails (server down, network error, timeout), it catches the exception and returns a friendly message instead of crashing. 
'''

@tool  # mandatory decorator: Registers the function as a tool the agent can offer to the model.
def check_server_status(server_url: str) -> str:  # Type hints
    """Check if a server is responding by making an HTTP request.

    Args:
        server_url: The URL of the server to check

    Returns:
        A message indicating whether the server is up or down
    """
    try:
        response = requests.get(server_url, timeout=5)
        return f"Server is up. Status code: {response.status_code}"
    except requests.exceptions.RequestException:
        return "Server is down or unreachable"

# ----------------------------------------

'''
Give the tool to the agent and Ask in plain English: "The bill is $85. What's a 20% tip, and how much does each person pay if we're splitting it 4 ways?"

I'll calculate the 20% tip and split the bill among 4 people for you.
Tool #1: calculate_tip # Strands streams the agent's words live, and prints a Tool #1: calculate_tip line the moment it runs your tool. 
Here's the breakdown:
- **Bill amount**: $85.00
- **20% tip**: $17.00
- **Total with tip**: $102.00
- **Amount per person** (splitting 4 ways): $25.50

So each person should pay $25.50.

Final answer: Here's the breakdown: …  So each person should pay $25.50.

'''


@tool # mandatory decorator: Registers the function as a tool the agent can offer to the model.
def calculate_tip(bill_amount: float, tip_percentage: float, num_people: int = 1) -> dict:
    """Calculate tip and split the bill among people.

    Args: # This is how the model learns that "20%" means tip_percentage=20, not 0.2.
        bill_amount: Total bill amount in dollars
        tip_percentage: Tip percentage (e.g., 15, 18, 20)
        num_people: Number of people splitting the bill (default: 1)

    Returns:
        A dictionary containing the bill, tip, total, and per person amount
    """
    tip = bill_amount * (tip_percentage / 100)
    total = bill_amount + tip
    per_person = total / num_people

    # A dictionary is fine. The agent hands the whole thing back to the model, which turns it into a sentence.
    return {
        "bill": bill_amount,
        "tip": round(tip, 2),
        "total": round(total, 2),
        "per_person": round(per_person, 2),
    }

# ----------------------------------------

'''
Imagine a sales assistant and the request: "Pull last quarter's sales data and email a summary to the team." 
That's not one task but three: query the database, analyse the numbers, send an email.
'''

@tool # mandatory decorator: Registers the function as a tool the agent can offer to the model.
def get_sales_data(quarter: str) -> dict:
    """Retrieve sales data for a specific quarter."""
    return {"revenue": 1250000, "deals": 47, "quarter": quarter}


@tool # mandatory decorator: Registers the function as a tool the agent can offer to the model.
def analyze_sales(revenue: int, deals: int, quarter: str) -> str:
    """Calculate key metrics from sales data."""
    avg_deal = revenue / deals
    return f"{quarter}: ${revenue:,} revenue, {deals} deals, ${avg_deal:,.0f} avg deal size"


@tool # mandatory decorator: Registers the function as a tool the agent can offer to the model.
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email message."""
    print(f"\n[EMAIL] to={to} | subject={subject}\n{body}\n")
    return f"Email sent to {to}"


# ----------------------------------------

# Use Duck Duck Go as Web Search Tool

from ddgs import DDGS

@tool # A Strands @tool that runs a DuckDuckGo text search. 
def websearch(keywords: str, region: str = "in-en", max_results: int = 5) -> str: # The default region in-en favours Indian English results. 
    # The docstring and Args become the tool description the model reads.
    """Search the web for up-to-date information.

    Args:
        keywords: The search query keywords.
        region: The search region, for example in-en, us-en, uk-en, wt-wt.
        max_results: The maximum number of results to return.
    """
    try:
        results = DDGS().text(keywords, region=region, max_results=max_results)
        return str(results) if results else "No results found."
    except Exception as e:
        return f"Exception: {e}"

# ----------------------------------------

@tool
def lookup_place(place: str) -> dict:
    """
    Look up a destination's prices.
    Returns: place, trip_type, room_price_per_night, travel_cost_per_person, daily_spend_per_person
    """
    # Simulated database (a real one arrives in lesson 07)
    db = {
        "GOA":    {"trip_type": "Beach", "room_price_per_night": 3000, "travel_cost_per_person": 2800, "daily_spend_per_person": 1500},
        "COORG":  {"trip_type": "Hills", "room_price_per_night": 2800, "travel_cost_per_person": 1000, "daily_spend_per_person": 1100},
        "MANALI": {"trip_type": "Hills", "room_price_per_night": 2500, "travel_cost_per_person": 9000, "daily_spend_per_person": 1300},
    }
    record = db.get(place.upper())
    if not record:
        return {"error": f"{place} not found"}
    return {"place": place.title(), **record}
    
    
# @tool
# def db_lookup(place: str) -> dict:
#     """Fetch a destination's trip type, best months, stay, room price, travel cost and daily spend from the database."""
#     record = fetch_place(place)
#     return record if record else {"error": f"{place} not found. Use list_destinations to see valid places."}
    
    
# @tool
# def list_destinations() -> list[dict]:
#     """Return all destinations in the database."""
#     return list_places()


# ----------------------------------------
@tool
def calculate_stay_share(room_price_per_night: float, nights: int, people: int) -> dict:
    """
    Calculate how much each person pays for the stay.
    One room fits 2 people, so rooms_needed = people / 2 rounded up.
    stay_share = rooms_needed * room_price_per_night * nights / people
    """
    rooms_needed = math.ceil(people / 2)
    stay_share = round(rooms_needed * room_price_per_night * nights / people, 2)
    return {"rooms_needed": rooms_needed, "stay_share": stay_share}


@tool
def calculate_trip_cost(travel_cost_per_person: float, stay_share: float,
                        daily_spend_per_person: float, nights: int) -> float:
    """
    Calculate the total trip cost for one person.
    total = travel_cost_per_person + stay_share + daily_spend_per_person * nights
    """
    return round(travel_cost_per_person + stay_share + daily_spend_per_person * nights, 2)


@tool
def trip_cost_tool(travel_cost_per_person: float, room_price_per_night: float,
                   daily_spend_per_person: float, people: int, nights: int) -> dict:
    """Compute rooms needed, stay share, food & fun, and total cost per person with exact arithmetic."""
    rooms_needed = math.ceil(people / 2)
    stay_share = round(rooms_needed * room_price_per_night * nights / people, 2)
    food_and_fun = round(daily_spend_per_person * nights, 2)
    total = round(travel_cost_per_person + stay_share + food_and_fun, 2)
    return {"rooms_needed": rooms_needed, "stay_share": stay_share,
            "food_and_fun": food_and_fun, "total_per_person": total}
            
# ── The calculator tool ────────────────────────────────────────────────────
@tool
def compute_trip_cost(
    travel_cost_per_person: float,
    room_price_per_night: float,
    daily_spend_per_person: float,
    people: int,
    nights: int,
    budget_per_person: float | None = None,
) -> dict:
    """
    Compute the trip cost per person with exact arithmetic.
    One room fits 2 people. Returns: rooms_needed, stay_share_per_person,
    food_and_fun_per_person, total_per_person, and (if a budget is given)
    within_budget and money_left_per_person (negative = over budget).
    """
    rooms_needed_per_person = math.ceil(people / 2)
    stay_share_per_person = round(rooms_needed_per_person * room_price_per_night * nights / people, 2)
    food_and_fun_per_person = round(daily_spend_per_person * nights, 2)
    total_per_person = round(travel_cost_per_person + stay_share_per_person + food_and_fun_per_person, 2)
    result = {
        "rooms_needed_per_person": rooms_needed_per_person,
        "stay_share_per_person": stay_share_per_person,
        "food_and_fun_per_person": food_and_fun_per_person,
        "total_per_person": total_per_person,
    }
    if budget_per_person is not None:
        result["within_budget"] = total_per_person <= budget_per_person
        result["money_left_per_person"] = round(budget_per_person - total_per_person, 2)
    return result
# ----------------------------------------
# Custom Tools creating WRITE updates 
# ----------------------------------------
# Custom Tools creating WRITE updates (Wrong approach in production if too many opening and closing of external i/o happens)
# This one simulates querying a product database.
# ----------------------------------------
# Custom Tools creating WRITE updates in production
# (prevent - multiple threads - race condition using lock) - using class level constructor thread-locking)
'''
Class tools: one shared connection:
----------------------------------
With class InventoryTools wrapper, What you want is to connect once when the agent starts and share that connection across all the tools. 
Separate decorated functions have no place to keep shared state, so the fix is to group related tools inside a class:


Connect once in `__init__`, share the connection through `self`.
Because the agent may call several tools at the same time, the shared connection is protected with a thread lock.
Always use parameterised queries (`?` placeholders) rather than f-strings inside SQL — the values come from the model, and you should treat them like user input.

'''

import threading  # not needed here — SQLiteDemoDB handles locking internally. Kept for the learning note below.

from local_demo_db.sqlite_demo_db import connect_to_database

class InventoryTools:
    def __init__(self):
        # connect_to_database() returns a thread-safe SQLiteDemoDB instance
        # that auto-seeds all *.csv files on first connect — no separate seed call needed.
        self.db = connect_to_database()

        # Threading note for learning:
        # SQLiteDemoDB already guards every query/execute with an internal
        # threading.Lock(), so consumer code does NOT need its own lock.
        # But if you were using a raw sqlite3.Connection instead, you'd do:
        #
        # self.lock = threading.Lock()
        #
        # and wrap every DB call with:
        #
        # with self.lock:
        #     row = self.db.execute("SELECT ...").fetchone()


    @tool # mandatory decorator: Registers the function as a tool the agent can offer to the model. Without it, this is just regular Python the agent never sees.
    def check_stock(self, product_id: str) -> str:
         # When someone asks "Is PROD-123 in stock?", the model reads "Check if a product is in stock" and recognises this tool fits.
        """Check product stock level.

        Args: # product_id … (e.g., "PROD-123") Tells the model which piece of the question to extract, and what it looks like.
            product_id: The product ID to check
        """

        # SQLiteDemoDB.query_one handles thread-locking internally
        row = self.db.query_one("SELECT name, quantity FROM inventory WHERE id = ?", (product_id,))

        # A plain sentence the model can use directly in its reply.
        return f"{row[0]} ({product_id}): {row[1]} units in stock" if row else f"Product {product_id} not found"


    @tool # mandatory decorator: Registers the function as a tool the agent can offer to the model. Without it, this is just regular Python the agent never sees.
    def update_stock(self, product_id: str, quantity: int) -> str:
        """Update product stock quantity.

        Args:
            product_id: The product ID to update
            quantity: New quantity to set
        """

        # SQLiteDemoDB.execute handles thread-locking + commit internally
        affected = self.db.execute("UPDATE inventory SET quantity = ? WHERE id = ?", (quantity, product_id))

        # A plain sentence the model can use directly in its reply.
        return f"Updated {product_id} to {quantity} units" if affected else f"Product {product_id} not found"


# ----------------------------------------
# Custom Tools only READ data
# ----------------------------------------
# (allow - multiple threads via concurrency) - using async + await inside tools along with invoke_async on parallel supported LLM models

# Non-Async - Non-Parallel - > Slow in production
@tool # mandatory decorator: Registers the function as a tool the agent can offer to the model. Without it, this is just regular Python the agent never sees.
def check_warehouse_inventory(product_id: str) -> str:
    # When someone asks "Is PROD-123 in stock?", the model reads "Check if a product is in stock" and recognises this tool fits.
    """Check if a product is in stock.

    Args: # product_id … (e.g., "PROD-123") Tells the model which piece of the question to extract, and what it looks like.
        product_id: The product ID to check (e.g., "PROD-123")
    """
    
    # A dictionary standing in for a database. In production you'd replace it with a real query or an API call to your inventory system; nothing else changes.
    inventory_db = {"PROD-123": 15, "PROD-456": 0, "PROD-789": 8}
    quantity = inventory_db.get(product_id, 0)

    if quantity > 0:
        return f"Product {product_id} is in stock. We have {quantity} units available."
        
    return f"Product {product_id} is currently out of stock."

# Total: 6 seconds of waiting
# warehouse_a = check_warehouse_inventory("PROD-123", "warehouse_a")  # 2 seconds
# warehouse_b = check_warehouse_inventory("PROD-123", "warehouse_b")  # 2 seconds
# warehouse_c = check_warehouse_inventory("PROD-123", "warehouse_c")  # 2 seconds
# ----------------------------------------
# Async - Parallel - > Recommended in production
# Make the tool `async` and use `await` inside it. When the model asks for all three checks parallely/concurrently in one turn, Strands runs them concurrently — about 2 seconds in total. Call the agent with `invoke_async`.

import asyncio # import httpx
import time

@tool # mandatory decorator: Registers the function as a tool the agent can offer to the model.
async def check_warehouse_inventory(product_id: str, warehouse: str) -> dict: # Marks the tool as pausable.
    """Check inventory at a specific warehouse.

    Args:
        product_id: Product ID to check
        warehouse: Warehouse identifier (e.g., "east", "west", "central")
    """
    
    # In a real tool you'd use an async HTTP client instead of asyncio.sleep
    # """Check inventory at a specific warehouse."""
    # async with httpx.AsyncClient() as client:
        # response 
        # = await client.get(f"https://api.warehouse-{warehouse}.example.com/inventory/{product_id}")
    # return response.json()["quantity"]
        
    await asyncio.sleep(2)  # prototype - locally simulate a slow API call - per connection ~ 2 secs # Stands in for a 2-second API call. While one call waits here, the others run.
    
    data = {
        "east": {"PROD-123": 45, "PROD-456": 12},
        "west": {"PROD-123": 30, "PROD-456": 0},
        "central": {"PROD-123": 60, "PROD-456": 25},
    }
    quantity = data.get(warehouse, {}).get(product_id, 0)
    return {"warehouse": warehouse, "product_id": product_id, "quantity": quantity} 

# async def main():
    # The async way to call an agent. When the model asks for all three checks in one turn, Strands runs them concurrently.
    # response = await agent.invoke_async("Can we ship 100 units of PROD-123? Check all warehouses: east, west, and central.")

# Works in Jupyter, which already has an event loop. In a plain .py script use asyncio.run(main()) instead.
# 45 + 30 + 60 = 135 units, so yes, 100 can ship. The tool waiting takes about 2 seconds instead of 6; total time also includes the model's thinking.
# await main()

# ----------------------------------------

