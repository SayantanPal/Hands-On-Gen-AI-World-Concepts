# =============================================================================
# MCP PROMPTS: a ready-made message the USER picks (like a slash command).
#    The LLM never calls a prompt by itself. MCP sends prompt arguments as text.
# =============================================================================
from capability.mcp_demo.demo_mcp_server.fast_mcp_server_obj import mcp

@mcp.prompt()
def plan_trip(place: str, people: str, nights: str, budget_per_person: str) -> str:
    """Plan a trip: fills the group's details into a well-phrased question."""
    return (
        f"We are {people} friends with ₹{budget_per_person} each. "
        f"Can we do {place} for {nights} nights? "
        "Give the cost per person, say whether it fits our budget, and add 2 useful tips."
    )

