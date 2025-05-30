import asyncio
import os
import json 
import traceback 
from typing import List, Dict, Any, Optional

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import BaseTool
from dotenv import load_dotenv

# Load environment variables from .env file in the ElderCare project root
_eldercare_project_root_absolute_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
load_dotenv(dotenv_path=os.path.join(_eldercare_project_root_absolute_path, ".env"))

class GmailService:
    def __init__(self):
        """
        Initializes the client for the Gmail MCP server.
        This service uses the MultiServerMCPClient to launch the npx command
        for the Gmail MCP server as a subprocess and communicate with it via stdio.
        """
        self.mcp_client = None
        self.tools: Optional[List[BaseTool]] = None
        self.tools_by_name: Dict[str, BaseTool] = {}
        self._loop = None

        try:
            self._loop = asyncio.get_event_loop_policy().get_event_loop()
            if self._loop.is_closed(): 
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
        except RuntimeError: 
             self._loop = asyncio.new_event_loop()
             asyncio.set_event_loop(self._loop)
        
        # Run the async initialization in the event loop
        self._loop.run_until_complete(self._initialize_client_and_tools())

    async def _initialize_client_and_tools(self):
        """
        Configures the MultiServerMCPClient and loads tools from the Gmail server.
        """
        self.client_config = {
            "gmail": {
                "command": "npx", # The command to run the MCP server
                "args": [
                    "@gongrzhe/server-gmail-autoauth-mcp"
                ],
                "transport": "stdio",
                # The auth keys are stored globally by the MCP's auth script (~/.gmail-mcp/),
                # so we don't need to pass them as env vars here.
            }
        }
        
        print("Initializing GmailService...")
        self.mcp_client = MultiServerMCPClient(self.client_config)
        
        try:
            print("Loading MCP tools for Gmail...")
            # Use the server name key from client_config
            self.tools = await self.mcp_client.get_tools(server_name="gmail") 
            if self.tools:
                self.tools_by_name = {tool.name: tool for tool in self.tools}
                print(f"Successfully loaded {len(self.tools)} Gmail tools:")
                for tool in self.tools: print(f"  - {tool.name}: {tool.description[:70]}...")
            else:
                print("WARNING: No tools loaded from Gmail MCP server.")
        except Exception as e:
            print(f"ERROR: Failed to load tools from Gmail MCP: {e}")
            traceback.print_exc()

    async def _invoke_tool_async(self, tool_name: str, tool_input: Dict[str, Any]) -> Any:
        if not self.mcp_client: return {"error": "MCP Client not initialized."}
        if tool_name not in self.tools_by_name: return {"error": f"Tool '{tool_name}' not found."}
        
        tool_to_invoke = self.tools_by_name[tool_name]
        print(f"Async invoking Gmail tool: {tool_name} with input: {json.dumps(tool_input, default=str)}")
        try:
            result = await tool_to_invoke.ainvoke(input=tool_input)
            print(f"Result from tool {tool_name}: {json.dumps(result, default=str)}")
            return result
        except Exception as e:
            print(f"ERROR: Exception during async tool invocation of '{tool_name}': {e}"); traceback.print_exc()
            return {"error": str(e)}

    def invoke_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> Any:
        if not self.tools_by_name: return {"error": "Tools not loaded."}
        if self._loop.is_closed(): self._loop = asyncio.new_event_loop(); asyncio.set_event_loop(self._loop)
        return self._loop.run_until_complete(self._invoke_tool_async(tool_name, tool_input))

    # --- Specific Tool Wrapper Methods ---
    # These wrappers correspond to the tools listed in the Gmail MCP README

    def send_email(self, to: List[str], subject: str, body: str, 
                   cc: Optional[List[str]] = None, bcc: Optional[List[str]] = None,
                   mimeType: str = "text/plain", htmlBody: Optional[str] = None) -> Optional[Any]:
        """Sends an email."""
        tool_input = {"to": to, "subject": subject, "body": body, "mimeType": mimeType}
        if cc: tool_input["cc"] = cc
        if bcc: tool_input["bcc"] = bcc
        if htmlBody: tool_input["htmlBody"] = htmlBody
        
        return self.invoke_tool(tool_name="send_email", tool_input=tool_input)

    def draft_email(self, to: List[str], subject: str, body: str, 
                    cc: Optional[List[str]] = None, bcc: Optional[List[str]] = None) -> Optional[Any]:
        """Creates a draft email."""
        tool_input = {"to": to, "subject": subject, "body": body}
        if cc: tool_input["cc"] = cc
        if bcc: tool_input["bcc"] = bcc
        
        return self.invoke_tool(tool_name="draft_email", tool_input=tool_input)

    def search_emails(self, query: str, max_results: int = 10) -> Optional[Any]:
        """Searches for emails."""
        tool_input = {"query": query, "maxResults": max_results}
        return self.invoke_tool(tool_name="search_emails", tool_input=tool_input)
    
    # ... You can add more wrapper methods for other tools like read_email, delete_email etc. as needed

    def close(self):
        print("Closing GmailService...")
        if self.mcp_client and hasattr(self.mcp_client, 'aclose'): 
             try: self._loop.run_until_complete(self.mcp_client.aclose()); print("Gmail MultiServerMCPClient aclosed.")
             except Exception as e: print(f"Error during Gmail mcp_client.aclose(): {e}")
        if self._loop and not self._loop.is_closed(): print("Closing asyncio event loop."); self._loop.close()


if __name__ == "__main__":
    print("Testing Gmail Service with langchain-mcp-adapters...")
    service = None
    try:
        # Before running, ensure you have authenticated the Gmail MCP server once
        # using 'npx @gongrzhe/server-gmail-autoauth-mcp auth'
        service = GmailService() 

        if service.tools:
            print(f"Available tools: {list(service.tools_by_name.keys())}") 

            print("\n--- Test 1: Search for emails (example) ---")
            # Using a generic query that should return something from most inboxes
            search_query = "from:google" 
            search_results = service.search_emails(query=search_query, max_results=2)
            print(f"Search results for query '{search_query}': {json.dumps(search_results, indent=2)}")

            print("\n--- Test 2: Draft an email (example) ---")
            # Replace with a real email address you can check for the draft
            DRAFT_RECIPIENT = os.getenv("TEST_EMAIL_RECIPIENT", "your-test-email@example.com")
            if "your-test-email" in DRAFT_RECIPIENT:
                print("Skipping draft email test: Please set TEST_EMAIL_RECIPIENT in your .env file.")
            else:
                draft_status = service.draft_email(
                    to=[DRAFT_RECIPIENT],
                    subject="Athena Test Draft",
                    body="This is a test draft created by the Athena agent."
                )
                print(f"Draft status: {json.dumps(draft_status, indent=2)}")
                print(f"Please check the 'Drafts' folder in your dummy Gmail account to verify.")
        else:
            print("Gmail Service tools not loaded. Aborting tests.")

    except Exception as e:
        print(f"An unexpected error occurred during testing: {e}"); traceback.print_exc()
    finally:
        if service:
            service.close()
    
    print("\nGmail Service testing finished.")

