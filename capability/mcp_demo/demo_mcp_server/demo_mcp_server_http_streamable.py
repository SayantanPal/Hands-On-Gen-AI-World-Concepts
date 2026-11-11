"""
The Demo Trip Buddy MCP server. It holds the database and offers it to any AI app through MCP.

The story
    Earlier, the tools lived inside the agent's own file. 
    Now we move them into a separate MCP server. 
    Any MCP-compatible app (our lesson 10 agent, Claude Desktop, Cursor, VS Code...) can now plug in and use Trip Buddy's data without custom code.

What it offers (the 3 MCP building blocks)
    Tools      the LLM decides when to call them      @mcp.tool()
        db_lookup, list_destinations, compute_trip_cost   (same 3 tools)
    Resources  read-only data at a URI; the APP decides when to read them   @mcp.resource()
        trips://destinations          all places with trip type and best months
        trips://place/{place_id}      one place's full record (a URI template)
    Prompts    a ready-made message the USER picks    @mcp.prompt()
        plan_trip(place, people, nights, budget_per_person)

How to run
    If you run it directly (uv run --active python capability/mcp_demo/demo_mcp_server/demo_mcp_server.py) it just waits silently for an MCP client. Press Ctrl+C to stop.
    
"""

from capability.mcp_demo.demo_mcp_server.fast_mcp_server_obj import mcp

# Importing these modules runs their @mcp.tool/@mcp.resource/@mcp.prompt
# decorators, which is what REGISTERS them on `mcp`. Without these imports the
# server starts with zero tools/resources/prompts.
from capability.mcp_demo.demo_mcp_server import (  # noqa: F401
    demo_mcp_server_tools,
    demo_mcp_server_resource,
    demo_mcp_server_prompt,
)

if __name__ == "__main__":
    mcp.run(transport="streamable-http")   # binds 127.0.0.1:8000 by default
