# app/services/spotify_service.py

import asyncio
import os
import json 
import traceback 
import re 
from typing import List, Dict, Any, Optional
from pathlib import Path # Import Path

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import BaseTool
from dotenv import load_dotenv

# Load environment variables from .env file in the ElderCare project root
_eldercare_project_root_absolute_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
load_dotenv(dotenv_path=os.path.join(_eldercare_project_root_absolute_path, ".env"))

# --- Configuration for the Spotify MCP (generic name) ---
SPOTIFY_MCP_PROJECT_ROOT = os.getenv("SPOTIFY_MCP_PROJECT_PATH")
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
SPOTIFY_REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI")


class SpotifyService:
    def __init__(self):
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
        
        self._loop.run_until_complete(self._initialize_client_and_tools())

    async def _initialize_client_and_tools(self):
        if not all([SPOTIFY_MCP_PROJECT_ROOT, SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, SPOTIFY_REDIRECT_URI]):
            print("ERROR: Missing necessary environment variables for Spotify MCP:")
            if not SPOTIFY_MCP_PROJECT_ROOT: print(" - SPOTIFY_MCP_PROJECT_PATH is not set in .env")
            if not SPOTIFY_CLIENT_ID: print(" - SPOTIFY_CLIENT_ID is not set in .env")
            if not SPOTIFY_CLIENT_SECRET: print(" - SPOTIFY_CLIENT_SECRET is not set in .env")
            if not SPOTIFY_REDIRECT_URI: print(" - SPOTIFY_REDIRECT_URI is not set in .env")
            return

        if not os.path.isdir(SPOTIFY_MCP_PROJECT_ROOT):
            print(f"FATAL ERROR: Spotify MCP project root not found at: {SPOTIFY_MCP_PROJECT_ROOT}")
            print("Please set SPOTIFY_MCP_PROJECT_PATH environment variable in your .env file to the root of the 'varunneal/spotify-mcp' clone.")
            return

        # --- Ensure .cache directory exists for Spotipy token caching ---
        spotipy_cache_dir = Path(SPOTIFY_MCP_PROJECT_ROOT) / ".cache"
        try:
            spotipy_cache_dir.mkdir(parents=True, exist_ok=True)
            print(f"Ensured Spotipy cache directory exists: {spotipy_cache_dir}")
        except Exception as e:
            print(f"Warning: Could not create Spotipy cache directory at {spotipy_cache_dir}: {e}")
        # --- End cache directory creation ---


        self.client_config = {
            "spotify_mcp_service": { 
                "command": "uv", 
                "args": ["--directory", SPOTIFY_MCP_PROJECT_ROOT, "run", "spotify-mcp"],
                "transport": "stdio",
                "env": { 
                    "SPOTIFY_CLIENT_ID": SPOTIFY_CLIENT_ID,
                    "SPOTIFY_CLIENT_SECRET": SPOTIFY_CLIENT_SECRET,
                    "SPOTIFY_REDIRECT_URI": SPOTIFY_REDIRECT_URI,
                    # Spotipy should now find and be able to write to this .cache directory
                    "XDG_CACHE_HOME": str(spotipy_cache_dir.parent), # Pass the parent of .cache as XDG_CACHE_HOME
                    "HOME": os.path.expanduser("~") 
                }
            }
        }
        
        print(f"Initializing SpotifyService with generic Spotify MCP.")
        print(f"MCP Project Root (used for --directory): {SPOTIFY_MCP_PROJECT_ROOT}")
        print(f"Using Redirect URI: {SPOTIFY_REDIRECT_URI}")
        
        self.mcp_client = MultiServerMCPClient(self.client_config)
        
        try:
            print("Loading MCP tools for Spotify...")
            self.tools = await self.mcp_client.get_tools(server_name="spotify_mcp_service") 
            if self.tools:
                self.tools_by_name = {tool.name: tool for tool in self.tools}
                print(f"Successfully loaded {len(self.tools)} Spotify tools:")
                for tool in self.tools: print(f"  - {tool.name}: {tool.description[:70]}...")
            else:
                print("WARNING: No tools loaded from Spotify MCP server.")
        except Exception as e:
            print(f"ERROR: Failed to load tools from Spotify MCP: {e}")
            traceback.print_exc()

    async def _invoke_tool_async(self, tool_name: str, tool_input: Dict[str, Any]) -> Any: 
        if not self.mcp_client: return {"error": "MCP Client not initialized."}
        if tool_name not in self.tools_by_name: return {"error": f"Tool '{tool_name}' not found."}
        
        tool_to_invoke = self.tools_by_name[tool_name]
        print(f"Async invoking tool: {tool_name} with input: {json.dumps(tool_input, default=str)}")
        try:
            result = await tool_to_invoke.ainvoke(input=tool_input) 
            if isinstance(result, str):
                print(f"Raw string result from tool {tool_name}: {result}")
                try:
                    parsed_result = json.loads(result)
                    print(f"Parsed JSON result from tool {tool_name}: {json.dumps(parsed_result, default=str)}")
                    return parsed_result
                except json.JSONDecodeError:
                    print(f"Result from tool {tool_name} is a string but not valid JSON. Returning as is.")
                    return result 
            else: 
                print(f"Result from tool {tool_name}: {json.dumps(result, default=str)}")
                return result
        except Exception as e:
            print(f"ERROR: Exception during async tool invocation of '{tool_name}': {e}"); traceback.print_exc()
            return {"error": str(e)}

    def invoke_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> Any: 
        if not self.tools_by_name: return {"error": "Tools not loaded."}
        if self._loop.is_closed(): self._loop = asyncio.new_event_loop(); asyncio.set_event_loop(self._loop)
        return self._loop.run_until_complete(self._invoke_tool_async(tool_name, tool_input))

    def search_spotify(self, query: str, item_type: str, limit: int = 5) -> Optional[Any]:
        tool_input = {"query": query, "qtype": item_type, "limit": limit}
        result = self.invoke_tool(tool_name="SpotifySearch", tool_input=tool_input) 
        if isinstance(result, dict) and "error" in result: 
            print(f"Error in search_spotify: {result['error']}")
            return None
        return result 

    def get_now_playing(self) -> Optional[dict]:
        tool_input = {"action": "get"}
        result = self.invoke_tool(tool_name="SpotifyPlayback", tool_input=tool_input)
        if isinstance(result, dict) and "error" in result: 
            print(f"Error in get_now_playing: {result['error']}")
            return None
        if isinstance(result, str): 
            try: return json.loads(result)
            except json.JSONDecodeError: print(f"get_now_playing: could not parse result string: {result}"); return None
        return result if isinstance(result, dict) else None

    def play_music(self, spotify_uri: Optional[str] = None, device_id: Optional[str] = None) -> Optional[Any]:
        if not spotify_uri:
            return {"error": "spotify_uri is required to play music."}
        tool_input = {"action": "start", "spotify_uri": spotify_uri}
        if device_id:
            print("Warning: device_id provided to play_music but not directly supported by 'SpotifyPlayback' tool's 'start' action schema in this MCP. Playback will be on default/active device.")
        return self.invoke_tool(tool_name="SpotifyPlayback", tool_input=tool_input)

    def pause_playback(self) -> Optional[Any]: 
        tool_input = {"action": "pause"}
        return self.invoke_tool(tool_name="SpotifyPlayback", tool_input=tool_input)

    def skip_track(self, num_skips: int = 1) -> Optional[Any]:
        tool_input = {"action": "skip", "num_skips": num_skips}
        return self.invoke_tool(tool_name="SpotifyPlayback", tool_input=tool_input)

    def get_queue(self) -> Optional[Any]:
        tool_input = {"action": "get"}
        result = self.invoke_tool(tool_name="SpotifyQueue", tool_input=tool_input)
        if isinstance(result, str): 
            try: return json.loads(result)
            except json.JSONDecodeError: print(f"get_queue: could not parse result string: {result}"); return None
        return result if isinstance(result, dict) else None

    def add_to_queue(self, track_id: str) -> Optional[Any]:
        if not track_id:
            return {"error": "track_id is required to add to queue."}
        tool_input = {"action": "add", "track_id": track_id}
        return self.invoke_tool(tool_name="SpotifyQueue", tool_input=tool_input)

    def get_item_info(self, item_uri: str) -> Optional[Any]:
        if not item_uri:
            return {"error": "item_uri is required to get info."}
        tool_input = {"item_uri": item_uri}
        result = self.invoke_tool(tool_name="SpotifyGetInfo", tool_input=tool_input)
        if isinstance(result, str): 
            try: return json.loads(result)
            except json.JSONDecodeError: print(f"get_item_info: could not parse result string: {result}"); return None
        return result if isinstance(result, dict) else None

    def close(self):
        print("Closing SpotifyService (Spotify MCP)...")
        if self.mcp_client and hasattr(self.mcp_client, 'aclose'): 
             try: self._loop.run_until_complete(self.mcp_client.aclose()); print("MultiServerMCPClient aclosed.")
             except Exception as e: print(f"Error during mcp_client.aclose(): {e}")
        if self._loop and not self._loop.is_closed(): print("Closing asyncio event loop."); self._loop.close()


if __name__ == "__main__":
    print("Testing Spotify Service with generic Spotify MCP...")
    import time
    service = None
    try:
        service = SpotifyService() 

        if service.tools:
            print(f"Available tools: {list(service.tools_by_name.keys())}") 

            print("\n--- Test 1: Search for a track ---")
            search_results = service.search_spotify(query="Bohemian Rhapsody", item_type="track", limit=1)
            track_uri_to_play = None
            track_name_to_play = None
            
            print(f"Search Results (raw from service): {json.dumps(search_results, indent=2)}")

            if isinstance(search_results, dict) and "tracks" in search_results:
                tracks_list = search_results.get("tracks")
                if isinstance(tracks_list, list) and tracks_list:
                    first_track = tracks_list[0] 
                    if isinstance(first_track, dict):
                        track_id = first_track.get("id")
                        if track_id:
                            track_uri_to_play = f"spotify:track:{track_id}" 
                            track_name_to_play = first_track.get("name")
                            artists_list = first_track.get("artists", []) 
                            artist_name = artists_list[0] if artists_list and isinstance(artists_list[0], str) else "N/A"
                            print(f"Found track: {track_name_to_play} by {artist_name} - Constructed URI: {track_uri_to_play}")
            elif isinstance(search_results, dict) and search_results.get("error"): 
                 print(f"Search returned an error: {search_results['error']}")
            else:
                print(f"Search result format not recognized or empty. Received type: {type(search_results)}, Value: {search_results}")


            if track_uri_to_play:
                print(f"\n--- Test 2: Play the found track: {track_name_to_play} ---")
                play_status = service.play_music(spotify_uri=track_uri_to_play)
                print(f"Play status: {json.dumps(play_status, indent=2)}")

                if play_status and isinstance(play_status, str) and "Playback starting" in play_status: 
                    print("Play command sent. Waiting...")
                    time.sleep(5) 
                    
                    print("\n--- Test 3: Get Now Playing ---")
                    now_playing = service.get_now_playing() 
                    if now_playing and isinstance(now_playing, dict) and now_playing.get('name'): 
                        artists_playing_list = now_playing.get("artists", [])
                        artist_playing_name = artists_playing_list[0] if artists_playing_list and isinstance(artists_playing_list[0], str) else "N/A"
                        print(f"Now Playing: {now_playing.get('name')} by {artist_playing_name}")
                    elif now_playing and isinstance(now_playing, str) and "No track playing" in now_playing:
                        print("No track currently playing.")
                    else: print(f"Could not get now playing info. Response: {now_playing}")
                    
                    time.sleep(2)
                    print("\n--- Test 4: Pause Playback ---")
                    pause_status = service.pause_playback() 
                    print(f"Pause status: {json.dumps(pause_status, indent=2)}")
                else: 
                    print(f"Could not play track or play command failed. Status: {play_status}")
            else: 
                print("Skipping play tests as no track URI was found from search.")
        else:
            print("Spotify Service tools not loaded. Aborting tests.")

    except FileNotFoundError as e:
        print(f"FATAL FileNotFoundError: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during testing: {e}"); traceback.print_exc()
    finally:
        if service: service.close()
    
    print("\nSpotify Service testing finished.")

