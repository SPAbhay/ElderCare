from typing import TypedDict, Annotated, List, Dict, Any, Optional, Tuple
import operator
import json
import traceback 
import re 
import os 

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage 
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langgraph.graph import StateGraph, END
from sqlalchemy.orm import Session
from pydantic import BaseModel, ValidationError 

# Project-specific imports
from app.core.llm_config import get_llm 
from app.prompts.prompt_templates import (
    ROUTING_PROMPT, 
    GENERIC_ENTITY_EXTRACTION_PROMPT, 
    ATHENA_SYSTEM_PROMPT,
    CHAT_SPOTIFY_ACTION_PARAM_EXTRACTION_PROMPT, 
    CHAT_GMAIL_SEND_PARAM_EXTRACTION_PROMPT,
    CHAT_GMAIL_SEARCH_PARAM_EXTRACTION_PROMPT,
    CHAT_GMAIL_READ_PARAM_EXTRACTION_PROMPT 
)
from app.crud import crud_user, crud_entity 
from app.core.database import SessionLocal 
from app.utils.temporal_parser import interpret_temporal_term 
from app.services.spotify_service import SpotifyService 
from app.services.gmail_service import GmailService

# --- Agent State Definition ---
class AgentState(TypedDict): 
    input: str                              
    chat_history: Annotated[List[BaseMessage], operator.add] 
    user_id: int                            
    user_profile: Optional[dict]            
    decision_outcome: Optional[str]         
    tool_parameters: Optional[dict] 
    tool_result: Optional[Any] 
    extracted_entities_json: Optional[list] 
    retrieved_facts_for_query: Optional[List[dict]] 
    response: Optional[str]                 
    error_message: Optional[str]            
    last_email_search_results: Optional[List[Dict[str, Any]]]

# --- Pydantic models for Parameter Validation ---
class ParsedQuery(BaseModel): 
    query_entity_type: Optional[str] = None
    query_identifier: Optional[str] = None
    query_attributes: Optional[List[str]] = None

class SpotifyActionParameters(BaseModel): 
    action: str 
    song_title: Optional[str] = None 
    artist_name: Optional[str] = None

class GmailSendParameters(BaseModel):
    to: List[str]
    subject: Optional[str] = None
    body: Optional[str] = None

class GmailSearchParameters(BaseModel):
    query: str

class GmailReadParameters(BaseModel):
    message_id: Optional[str] = None
    position: Optional[int] = None
    
class GmailReadParameters(BaseModel): 
    message_id: Optional[str] = None

# --- LLM Initialization ---
USER_SPECIFIED_DEFAULT_MODEL = os.getenv("DEFAULT_OPENROUTER_MODEL", "qwen/qwen3-235b-a22b:free")
ROUTING_MODEL_NAME = os.getenv("ROUTING_MODEL_NAME", "meta-llama/llama-3.3-70b-instruct:free")
FACT_EXTRACTION_MODEL_NAME = os.getenv("FACT_EXTRACTION_MODEL_NAME", "meta-llama/llama-3.3-70b-instruct:free")
QUERY_UNDERSTANDING_MODEL_NAME = os.getenv("QUERY_UNDERSTANDING_MODEL_NAME", USER_SPECIFIED_DEFAULT_MODEL) 
CONVERSATIONAL_MODEL_NAME = os.getenv("CONVERSATIONAL_MODEL_NAME", "meta-llama/llama-3.3-70b-instruct:free") 
TOOL_PARAM_EXTRACTION_MODEL_NAME = os.getenv("TOOL_PARAM_EXTRACTION_MODEL_NAME", "meta-llama/llama-3.3-70b-instruct:free") # Used for Spotify action params

print(f"Using ROUTING_MODEL_NAME: {ROUTING_MODEL_NAME}")
print(f"Using FACT_EXTRACTION_MODEL_NAME: {FACT_EXTRACTION_MODEL_NAME}")
print(f"Using QUERY_UNDERSTANDING_MODEL_NAME: {QUERY_UNDERSTANDING_MODEL_NAME}")
print(f"Using CONVERSATIONAL_MODEL_NAME: {CONVERSATIONAL_MODEL_NAME}")
print(f"Using TOOL_PARAM_EXTRACTION_MODEL_NAME: {TOOL_PARAM_EXTRACTION_MODEL_NAME}")

fact_extraction_llm = get_llm(model_name=FACT_EXTRACTION_MODEL_NAME, temperature=0.1) 
routing_llm = get_llm(model_name=ROUTING_MODEL_NAME, temperature=0.0) 
query_understanding_llm = get_llm(model_name=QUERY_UNDERSTANDING_MODEL_NAME, temperature=0.0) 
conversational_llm = get_llm(model_name=CONVERSATIONAL_MODEL_NAME, temperature=0.7)
tool_param_extraction_llm = get_llm(model_name=TOOL_PARAM_EXTRACTION_MODEL_NAME, temperature=0.0)

# --- Initialize Services ---
try:
    spotify_service_instance = SpotifyService()
    print("SpotifyService initialized successfully.")
except Exception as e:
    print(f"Failed to initialize SpotifyService: {e}"); spotify_service_instance = None

try:
    gmail_service_instance = GmailService()
    print("GmailService initialized successfully.")
except Exception as e:
    print(f"Failed to initialize GmailService: {e}"); gmail_service_instance = None

# --- Node Definitions ---
def get_db_session(): 
    db = SessionLocal();  
    try: 
        yield db; 
    finally: 
        db.close()

def agent_decision_node(state: AgentState) -> AgentState:
    print("\n--- Node: Agent Decision ---")
    user_input = state["input"]; user_id = state["user_id"]
    node_error_message = None; decision = "generate_response" 
    print(f"Routing input: '{user_input}' for user_id: {user_id}")
    if not routing_llm: return {**state, "decision_outcome": decision, "error_message": "Routing LLM not configured."}
    try:
        db = next(get_db_session()); user_facts_for_prompt_str = "{}" 
        try:
            current_user = crud_user.get_user(db, user_id)
            if current_user: user_facts_for_prompt_str = json.dumps({"username": current_user.username, "location": current_user.location, "hobbies": current_user.hobbies_json, "jobs": current_user.job_json, "preferences": current_user.preferences_json})
        finally: db.close()
        
        prompt_messages = ROUTING_PROMPT.format_messages(user_facts=user_facts_for_prompt_str, input=user_input)
        llm_response = routing_llm.invoke(prompt_messages)
        raw_llm_output_for_routing = llm_response.content.strip()
        print(f"LLM raw output for routing: '{raw_llm_output_for_routing}'")
        extracted_decision_keyword = raw_llm_output_for_routing.split("</think>")[-1].strip().split(maxsplit=1)[0].lower() if "</think>" in raw_llm_output_for_routing else raw_llm_output_for_routing.split(maxsplit=1)[0].lower()
        print(f"Cleaned decision keyword: '{extracted_decision_keyword}'")
        
        allowed_decisions = ["extract_facts", "query_facts", "generate_response", "exit", 
                             "spotify_playback_action", "gmail_send_email", 
                             "gmail_search_emails", "gmail_read_email"] 
        if extracted_decision_keyword in allowed_decisions: decision = extracted_decision_keyword
        else: node_error_message = f"Routing LLM returned unexpected decision: '{extracted_decision_keyword}'"
    except Exception as e: traceback.print_exc(); node_error_message = f"Core error in agent_decision_node: {e}"
    print(f"Final Decision: {decision}")
    return {**state, "decision_outcome": decision, "tool_parameters": None, "tool_result": None, "retrieved_facts_for_query": [] if decision != "query_facts" else state.get("retrieved_facts_for_query"), "error_message": node_error_message or state.get("error_message")}

def extract_general_facts_node(state: AgentState) -> AgentState: 
    print("\n--- Node: Extract General Facts ---")
    if state.get("decision_outcome") != "extract_facts":
        return {**state, "extracted_entities_json": []} 
    user_input = state["input"]; user_id = state["user_id"]
    processed_entities_for_state = []; node_error_message = None
    print(f"Attempting to extract general facts from input: '{user_input}' for user_id: {user_id}")
    if not fact_extraction_llm: return {**state, "error_message": "Fact LLM not configured.", "extracted_entities_json": []}
    try:
        prompt_messages = GENERIC_ENTITY_EXTRACTION_PROMPT.format_messages(input=user_input)
        llm_response = fact_extraction_llm.invoke(prompt_messages)
        llm_output_json_str = llm_response.content
        print(f"DEBUG (extract_general_facts_node): LLM raw output:\n<<<\n{llm_output_json_str}\n>>>")
        
        potential_json = None; parsed_output = None; identified_entities = []
        try:
            stripped_llm_output = re.sub(r"<think>.*?</think>", "", llm_output_json_str, flags=re.DOTALL).strip()
            match = re.search(r"```json\s*(\{.*?\})\s*```", stripped_llm_output, re.DOTALL) or re.search(r"(\{.*?\})", stripped_llm_output, re.DOTALL)
            if match:
                potential_json = match.group(1).strip()
                decoder = json.JSONDecoder(); parsed_output, _ = decoder.raw_decode(potential_json)
                identified_entities = parsed_output.get("identified_entities", [])
                if not isinstance(identified_entities, list): node_error_message = "LLM output for general entities was not a list."; identified_entities = []
            else: node_error_message = "No JSON object pattern found in LLM output for general fact extraction."
        except json.JSONDecodeError as e_parse: node_error_message = f"Parse error: {e_parse}"
        except Exception as e_parse_other: node_error_message = f"Unexpected parse error: {e_parse_other}"; traceback.print_exc()

        if identified_entities:
            print(f"DEBUG (extract_general_facts_node): LLM identified {len(identified_entities)} entities. Attempting to write to DB...")
            db = next(get_db_session())
            try:
                for entity_data in identified_entities:
                    if isinstance(entity_data, dict) and "entity_type" in entity_data and "details" in entity_data:
                        # ... temporal interpretation ...
                        print(f"DEBUG (extract_general_facts_node): Writing entity - Type: {entity_data['entity_type']}, Details: {json.dumps(entity_data['details'])}")
                        created_entity = crud_entity.create_user_entity(db, user_id, entity_data["entity_type"], entity_data["details"])
                        if created_entity: 
                            processed_entities_for_state.append({"id": created_entity.id, "entity_type": created_entity.entity_type, "details_json": created_entity.details_json})
                            print(f"DEBUG (extract_general_facts_node): Successfully saved entity with new ID: {created_entity.id}")
            finally: db.close()
    except Exception as e: node_error_message = f"Core error in extract_general_facts_node: {e}"; traceback.print_exc()
    return {**state, "extracted_entities_json": processed_entities_for_state, "error_message": node_error_message or state.get("error_message")}

def _extract_parameters_with_llm(user_input: str, 
                                 prompt_template: ChatPromptTemplate, 
                                 pydantic_model: type[BaseModel], # Use type[BaseModel] for Pydantic model class
                                 node_name: str,
                                 additional_prompt_kwargs: Optional[Dict[str, Any]] = None
                                 ) -> Tuple[Optional[Dict[str,Any]], Optional[str]]:
    if not tool_param_extraction_llm:
        return None, f"{node_name}: Tool parameter extraction LLM not configured."
    try:
        format_kwargs = {"input": user_input}
        if additional_prompt_kwargs:
            format_kwargs.update(additional_prompt_kwargs)
        
        prompt_messages = prompt_template.format_messages(**format_kwargs)
        print(f"Sending prompt to LLM ({tool_param_extraction_llm.model_name if hasattr(tool_param_extraction_llm, 'model_name') else 'N/A'}) for {node_name} with kwargs: {list(format_kwargs.keys())}...")
        llm_response = tool_param_extraction_llm.invoke(prompt_messages)
        raw_json_output = llm_response.content.strip()
        print(f"DEBUG ({node_name}): LLM raw output for params:\n<<<\n{raw_json_output}\n>>>")
        
        potential_json_str = None
        stripped_llm_output = re.sub(r"<think>.*?</think>", "", raw_json_output, flags=re.DOTALL).strip()
        match = re.search(r"```json\s*(\{.*?\})\s*```", stripped_llm_output, re.DOTALL) or re.search(r"(\{.*?\})", stripped_llm_output, re.DOTALL)
        
        if match:
            potential_json_str = match.group(1).strip()
            print(f"Attempting to parse extracted JSON block for {node_name}:\n<<<\n{potential_json_str}\n>>>")
            parsed_dict_from_llm, _ = json.JSONDecoder().raw_decode(potential_json_str)
            validated_params = pydantic_model.model_validate(parsed_dict_from_llm)
            print(f"DEBUG ({node_name}): Validated params: {validated_params.model_dump()}")
            return validated_params.model_dump(exclude_none=True), None
        return None, f"{node_name}: No JSON object found in LLM output."
    except (json.JSONDecodeError, ValidationError) as e_parse:
        return None, f"{node_name}: Failed to parse/validate JSON params: {e_parse}. Raw: {potential_json_str or raw_json_output}"
    except Exception as e:
        traceback.print_exc()
        return None, f"{node_name}: Core error: {e}"
    
def extract_spotify_action_parameters_node(state: AgentState) -> AgentState: # Renamed
    print("\n--- Node: Extract Spotify Action Parameters ---")
    user_input = state["input"]; node_error_message = None; extracted_params_dict: Optional[Dict[str, Any]] = None
    if not tool_param_extraction_llm: return {**state, "tool_parameters": None, "error_message": "Tool parameter extraction LLM not configured."}
    try:
        prompt_messages = CHAT_SPOTIFY_ACTION_PARAM_EXTRACTION_PROMPT.format_messages(input=user_input)
        print(f"Sending prompt to LLM ({tool_param_extraction_llm.model_name if hasattr(tool_param_extraction_llm, 'model_name') else 'N/A'}) for Spotify action parameters...")
        llm_response = tool_param_extraction_llm.invoke(prompt_messages) 
        raw_json_output = llm_response.content.strip()
        print(f"LLM raw output for Spotify action params:\n<<<\n{raw_json_output}\n>>>")
        potential_json_str = None; parsed_dict_from_llm = None
        try:
            stripped_llm_output = raw_json_output
            if "<think>" in stripped_llm_output: stripped_llm_output = re.sub(r"<think>.*?</think>", "", stripped_llm_output, flags=re.DOTALL).strip()
            match = re.search(r"```json\s*(\{.*?\})\s*```", stripped_llm_output, re.DOTALL)
            if match: potential_json_str = match.group(1).strip()
            else: 
                match = re.search(r"(\{.*?\})", stripped_llm_output, re.DOTALL) 
                if match: potential_json_str = match.group(1).strip()
            if potential_json_str:
                print(f"Attempting to parse extracted JSON block for Spotify action params:\n<<<\n{potential_json_str}\n>>>")
                decoder = json.JSONDecoder(); parsed_dict_from_llm, _ = decoder.raw_decode(potential_json_str)
                validated_params = SpotifyActionParameters.model_validate(parsed_dict_from_llm) # Use new Pydantic model
                extracted_params_dict = validated_params.model_dump(exclude_none=True)
            else: node_error_message = "No JSON object found in Spotify action parameter extraction output."
        except (json.JSONDecodeError, ValidationError) as e_parse:
            node_error_message = f"Failed to parse/validate JSON for Spotify action params: {e_parse}"
            print(f"Problematic JSON for Spotify action params: {potential_json_str or raw_json_output}")
        print(f"Extracted Spotify action parameters: {extracted_params_dict}")
    except Exception as e: traceback.print_exc(); node_error_message = f"Core error in extract_spotify_action_parameters_node: {e}"
    return {**state, "tool_parameters": extracted_params_dict, "error_message": node_error_message or state.get("error_message")}

def extract_gmail_send_parameters_node(state: AgentState) -> AgentState:
    print("\n--- Node: Extract Gmail Send Parameters ---")
    params, err = _extract_parameters_with_llm(state["input"], CHAT_GMAIL_SEND_PARAM_EXTRACTION_PROMPT, GmailSendParameters, "GmailSendParams")
    return {**state, "tool_parameters": params, "error_message": err or state.get("error_message")}

def extract_gmail_search_parameters_node(state: AgentState) -> AgentState:
    print("\n--- Node: Extract Gmail Search Parameters ---")
    params, err = _extract_parameters_with_llm(state["input"], CHAT_GMAIL_SEARCH_PARAM_EXTRACTION_PROMPT, GmailSearchParameters, "GmailSearchParams")
    return {**state, "tool_parameters": params, "error_message": err or state.get("error_message")}

def extract_gmail_read_parameters_node(state: AgentState) -> AgentState: # MODIFIED
    print("\n--- Node: Extract Gmail Read Parameters ---")
    user_input = state["input"]
    last_search_results = state.get("last_email_search_results")
    
    search_context_str = "No recent email search results available to reference by position."
    if last_search_results and isinstance(last_search_results, list):
        formatted_search_items = []
        for i, email_summary in enumerate(last_search_results[:5]): # Provide context for up to 5 emails
            if isinstance(email_summary, dict):
                subj = email_summary.get('subject', 'No Subject')
                from_sender = email_summary.get('from', 'Unknown Sender')
                msg_id = email_summary.get('id', 'Unknown ID') # Important: MCP returns 'id'
                formatted_search_items.append(f"{i+1}. ID: {msg_id}, Subject: '{subj}', From: {from_sender}")
        if formatted_search_items:
            search_context_str = "Context of recent email search results (user might refer to these by number):\n" + "\n".join(formatted_search_items)
            
    print(f"DEBUG (extract_gmail_read_parameters_node): Context for LLM: {search_context_str}")
    
    params, err = _extract_parameters_with_llm(
        user_input, 
        CHAT_GMAIL_READ_PARAM_EXTRACTION_PROMPT, 
        GmailReadParameters, 
        "GmailReadParams",
        additional_prompt_kwargs={"email_search_context": search_context_str}
    )
    return {**state, "tool_parameters": params, "error_message": err or state.get("error_message")}


def query_facts_node(state: AgentState) -> AgentState: 
    print("\n--- Node: Query Facts ---")
    user_input = state["input"]; user_id = state["user_id"]
    retrieved_facts_for_response = []; node_error_message = None
    print(f"Attempting to understand query: '{user_input}' for user_id: {user_id}")
    if not query_understanding_llm: return {**state, "retrieved_facts_for_query": [], "error_message": "Query LLM not configured."}
    try:
        query_parser_prompt_template = PromptTemplate(
            input_variables=["question", "entity_type_examples"],
            template="""
            Analyze the user's question: "{question}"
            What type of entity are they asking about? Choose from common types or infer a new one.
            Common entity types could be: {entity_type_examples}.
            What specific details or attributes are they interested in regarding that entity type?
            If they mention a name or identifier, extract that too.
            Respond ONLY with a JSON object with the following keys:
            "query_entity_type": "string (e.g., 'pet', 'event', 'personal_info', 'user_hobby')",
            "query_identifier": "string (e.g., name of pet/person, description of event, or null if general query for type)",
            "query_attributes": ["list of strings (e.g., ['breed', 'color'], ['date_text'], ['location'], or null if asking for all details of an identified entity)"]
            Example 1: Question: "What is my cat Whiskers' breed?" Output: {{"query_entity_type": "pet", "query_identifier": "Whiskers", "query_attributes": ["breed"]}}
            Example 2: Question: "Do you remember where I live?" Output: {{"query_entity_type": "personal_info", "query_identifier": null, "query_attributes": ["location"]}}
            Example 3: Question: "Tell me about my meeting next week." Output: {{"query_entity_type": "event", "query_identifier": "meeting", "query_attributes": null}} 
            Example 4: Question: "What are my hobbies?" Output: {{"query_entity_type": "user_hobby", "query_identifier": null, "query_attributes": null}}
            Question: "{question}" Output: """
        )
        common_entity_types = "personal_info, user_hobby, user_job, pet, family_member, event, etc."
        formatted_parser_prompt = query_parser_prompt_template.format(question=user_input, entity_type_examples=common_entity_types)
        llm_response = query_understanding_llm.invoke(formatted_parser_prompt) 
        raw_query_structure_str = llm_response.content if hasattr(llm_response, 'content') else llm_response
        print(f"DEBUG (query_facts_node): LLM raw output for query understanding:\n<<<\n{raw_query_structure_str}\n>>>")
        
        parsed_query_dict: Optional[Dict[str, Any]] = None; validated_query_structure: Optional[ParsedQuery] = None; potential_query_json_str = None 
        try:
            stripped_llm_output = raw_query_structure_str
            if "<think>" in stripped_llm_output: stripped_llm_output = re.sub(r"<think>.*?</think>", "", stripped_llm_output, flags=re.DOTALL).strip()
            match = re.search(r"```json\s*(\{.*?\})\s*```", stripped_llm_output, re.DOTALL)
            if match: potential_query_json_str = match.group(1).strip()
            else: 
                match = re.search(r"(\{.*?\})", stripped_llm_output, re.DOTALL) 
                if match: potential_query_json_str = match.group(1).strip()
            if potential_query_json_str:
                print(f"Attempting to parse extracted JSON block for query structure:\n<<<\n{potential_query_json_str}\n>>>")
                decoder = json.JSONDecoder(); parsed_query_dict, end_index = decoder.raw_decode(potential_query_json_str)
                try:
                    validated_query_structure = ParsedQuery.model_validate(parsed_query_dict)
                    print(f"Validated query structure (Pydantic): {validated_query_structure.model_dump()}")
                except ValidationError as e_pydantic: node_error_message = f"Query structure Pydantic validation failed: {e_pydantic}"
            else: node_error_message = "No JSON object pattern found in query LLM output."
        except json.JSONDecodeError as e_json:
            node_error_message = f"Query structure JSON parse error: {e_json}"
            print(f"Problematic query JSON string: {potential_query_json_str or raw_query_structure_str}")
        except Exception as e_parse: traceback.print_exc(); node_error_message = f"Unexpected query parse error: {e_parse}"

        if validated_query_structure: 
            query_entity_type = validated_query_structure.query_entity_type; query_identifier = validated_query_structure.query_identifier; query_attributes = validated_query_structure.query_attributes
            db = next(get_db_session())
            try:
                if query_entity_type in ["personal_info", "user_location", "user_hobby", "user_job", "user_preference_general"]:
                    user_profile_db = crud_user.get_user(db, user_id) 
                    if user_profile_db:
                        if query_entity_type in ["personal_info", "user_location"] and (not query_attributes or "location" in query_attributes) and user_profile_db.location: retrieved_facts_for_response.append({"type": "location", "detail": user_profile_db.location})
                        if query_entity_type == "user_hobby" and (not query_attributes or not query_attributes) and user_profile_db.hobbies_json: retrieved_facts_for_response.append({"type": "hobbies", "details": user_profile_db.hobbies_json}) 
                        if query_entity_type == "user_job" and (not query_attributes or not query_attributes) and user_profile_db.job_json: retrieved_facts_for_response.append({"type": "jobs", "details": user_profile_db.job_json}) 
                elif query_entity_type: 
                    entities_from_db = crud_entity.get_user_entities(db, user_id=user_id, entity_type=query_entity_type)
                    print(f"Found {len(entities_from_db)} entities of type '{query_entity_type}' for user {user_id}.")
                    for entity in entities_from_db:
                        details = entity.details_json
                        if not isinstance(details, dict): continue
                        entity_matches_identifier = True 
                        if query_identifier:
                            id_lower = str(query_identifier).lower()
                            name_match = str(details.get("name","")).lower() == id_lower; desc_match = str(details.get("description","")).lower() == id_lower; title_match = str(details.get("title","")).lower() == id_lower
                            date_text_match = str(details.get("date_text","")).lower().find(id_lower) != -1 if query_entity_type == "event" else False
                            entity_matches_identifier = name_match or desc_match or title_match or date_text_match
                        if entity_matches_identifier:
                            fact_detail_to_add = {"entity_type": entity.entity_type}
                            if query_attributes and isinstance(query_attributes, list) and len(query_attributes) > 0: 
                                specific_details = {}; [specific_details.update({attr: details[attr]}) for attr in query_attributes if attr in details]
                                if specific_details: fact_detail_to_add["details"] = specific_details
                                else: continue 
                            else: fact_detail_to_add["details"] = details
                            if "details" in fact_detail_to_add and fact_detail_to_add["details"]: retrieved_facts_for_response.append(fact_detail_to_add)
            finally: db.close()
        if not retrieved_facts_for_response and not node_error_message: print("No specific facts found for the query based on parsed structure.")
    except Exception as e: traceback.print_exc(); node_error_message = f"Core error in query_facts_node: {e}"
    print(f"Retrieved facts for query: {json.dumps(retrieved_facts_for_response, indent=2)}")
    return {**state, "retrieved_facts_for_query": retrieved_facts_for_response, "error_message": node_error_message or state.get("error_message")}

def spotify_action_node(state: AgentState) -> AgentState: # Renamed and MODIFIED
    print("\n--- Node: Spotify Action ---")
    tool_params = state.get("tool_parameters") # Expected: {"action": ..., "song_title": ..., "artist_name": ...}
    action_result = "Could not understand the Spotify request."
    node_error_message = state.get("error_message") 

    if not spotify_service_instance:
        return {**state, "tool_result": "Spotify service is not available.", "error_message": "Spotify service not initialized."}
    
    if not tool_params or not isinstance(tool_params, dict) or not tool_params.get("action"):
        node_error_message = node_error_message or "Spotify action parameters (especially 'action') not extracted."
        action_result = "I couldn't understand what you wanted to do with Spotify."
        print(f"Spotify Action Node: {node_error_message}. Tool params: {tool_params}")
    else:
        action = tool_params.get("action")
        song_title = tool_params.get("song_title")
        artist_name = tool_params.get("artist_name")
        
        if action == "start":
            if not song_title: # song_title is now expected to be set by param extraction, even for moods/genres
                action_result = "I need a song title, artist, or mood/genre to start playing."
                node_error_message = "Missing song title for 'start' action after parameter extraction."
            else:
                search_query = song_title
                if artist_name: # This artist might be from user or suggested by LLM
                    search_query += f" artist:{artist_name}"
                
                print(f"Searching Spotify for: '{search_query}' to play.")
                search_results = spotify_service_instance.search_spotify(query=search_query, item_type="track", limit=1)
                
                track_uri_to_play = None; found_track_name = None; found_artist_name_from_search = None
                if isinstance(search_results, dict) and "tracks" in search_results:
                    tracks_list = search_results.get("tracks")
                    if isinstance(tracks_list, list) and tracks_list and isinstance(tracks_list[0], dict):
                        first_track = tracks_list[0]
                        track_id = first_track.get("id")
                        if track_id:
                            track_uri_to_play = f"spotify:track:{track_id}"
                            found_track_name = first_track.get("name")
                            artists = first_track.get("artists", ["Unknown Artist"])
                            found_artist_name_from_search = artists[0] if artists and isinstance(artists[0], str) else "Unknown Artist"
                elif isinstance(search_results, dict) and search_results.get("error"):
                    action_result = f"Error searching Spotify: {search_results['error']}"; node_error_message = action_result
                
                if track_uri_to_play:
                    print(f"Found track: '{found_track_name}' by {found_artist_name_from_search}. URI: {track_uri_to_play}. Attempting to play...")
                    play_status = spotify_service_instance.play_music(spotify_uri=track_uri_to_play)
                    if play_status and isinstance(play_status, str) and "playback starting" in play_status.lower() or play_status is None:
                        action_result = f"Now playing '{found_track_name}' by {found_artist_name_from_search}' on Spotify."
                        node_error_message = None
                    else:
                        error_detail = play_status.get("error") if isinstance(play_status, dict) else play_status
                        action_result = f"I found '{found_track_name}', but couldn't play it. Spotify said: {error_detail}"
                        node_error_message = action_result
                elif not node_error_message:
                    action_result = f"Sorry, I couldn't find '{search_query}' on Spotify to play."
                    node_error_message = "Song not found for playback."
        elif action == "pause":
            print("Attempting to pause playback via Spotify service...")
            pause_status = spotify_service_instance.pause_playback()
            print(f"Pause status from service: {pause_status}")
            if pause_status and isinstance(pause_status, str) and "playback paused" in pause_status.lower() or pause_status is None:
                action_result = "Playback paused on Spotify."
                node_error_message = None
            else: 
                error_detail = pause_status.get("error") if isinstance(pause_status, dict) else pause_status
                action_result = f"Could not pause playback. Spotify said: {error_detail}"
                node_error_message = action_result
        elif action == "skip":
            print("Attempting to skip track via Spotify service...")
            skip_status = spotify_service_instance.skip_track() 
            if skip_status and isinstance(skip_status, str) and ("skipped" in skip_status.lower() or "next track" in skip_status.lower()) or skip_status is None:
                action_result = "Skipped to the next track on Spotify."
                node_error_message = None
            else:
                error_detail = skip_status.get("error") if isinstance(skip_status, dict) else skip_status
                action_result = f"Could not skip track. Spotify said: {error_detail}"
                node_error_message = action_result
        elif action == "get": # Get current playing song
            print("Attempting to get current playing song via Spotify service...")
            now_playing = spotify_service_instance.get_now_playing()
            if now_playing and isinstance(now_playing, dict) and now_playing.get('name'):
                artists_playing = now_playing.get("artists", ["Unknown Artist"])
                artist_playing = artists_playing[0] if artists_playing and isinstance(artists_playing[0], str) else "Unknown Artist"
                action_result = f"Currently playing '{now_playing.get('name')}' by {artist_playing}."
                node_error_message = None
            elif now_playing and isinstance(now_playing, str) and "no track playing" in now_playing.lower():
                action_result = "Nothing is currently playing on Spotify."
                node_error_message = None
            else:
                action_result = "Could not get current playing information from Spotify."
                node_error_message = f"Error getting current track: {now_playing}"
        else:
            action_result = f"I'm not sure how to perform the Spotify action: '{action}'."
            node_error_message = f"Unknown Spotify action: {action}"

    print(f"Spotify Action Node Result: {action_result}")
    return {**state, "tool_result": action_result, "error_message": node_error_message}

def gmail_send_action_node(state: AgentState) -> AgentState: # MODIFIED
    print("\n--- Node: Gmail Send Action ---")
    params = state.get("tool_parameters")
    action_result = "Could not prepare email."; node_error_message = state.get("error_message")

    if not gmail_service_instance:
        return {**state, "tool_result": "Gmail service unavailable.", "error_message": "Gmail service not initialized."}
    
    if not params or not isinstance(params, dict):
        node_error_message = node_error_message or "Could not understand email details."
        action_result = "I'm sorry, I couldn't quite understand the details for the email."
    else:
        # --- Parameter Validation (Slot Filling Logic) ---
        recipient = params.get("to")
        subject = params.get("subject")
        body = params.get("body")

        if not recipient or not recipient[0]:
            action_result = "I'm ready to send an email, but I need to know who it's for. Who should I send it to?"
            node_error_message = "Missing recipient."
        elif not subject:
            action_result = f"Okay, I can send an email to {recipient[0]}. What should the subject line be?"
            node_error_message = "Missing subject."
        elif not body:
            action_result = f"Okay, I'm sending an email to {recipient[0]} with the subject '{subject}'. What would you like the body of the email to say?"
            node_error_message = "Missing email body."
        else:
            # All parameters are present, proceed to send
            print(f"Attempting to send email with params: {params}")
            send_status = gmail_service_instance.send_email(to=recipient, subject=subject, body=body)
            
            if send_status and not (isinstance(send_status, dict) and send_status.get("error")):
                action_result = f"Email to {', '.join(recipient)} has been sent successfully."
                node_error_message = None # Clear previous errors on success
            else:
                error_detail = send_status.get("error") if isinstance(send_status, dict) else send_status
                action_result = f"Sorry, I couldn't send the email. The service said: {error_detail}"
                node_error_message = action_result
    
    return {**state, "tool_result": action_result, "error_message": node_error_message}

def gmail_search_action_node(state: AgentState) -> AgentState: # MODIFIED
    print("\n--- Node: Gmail Search Action ---")
    params = state.get("tool_parameters")
    action_result_summary = "Could not understand search query."; node_error_message = state.get("error_message")
    parsed_email_summaries: List[Dict[str, Any]] = [] 

    if not gmail_service_instance: 
        return {**state, "tool_result": "Gmail service unavailable.", "error_message": "Gmail service not initialized.", "last_email_search_results": None}
    if not params or not isinstance(params.get("query"), str):
        node_error_message = node_error_message or "Missing search query for Gmail."
        action_result_summary = "I need a search query to find emails."
    else:
        print(f"Searching Gmail with query: '{params['query']}'")
        search_results_from_service = gmail_service_instance.search_emails(query=params["query"], max_results=5) 
        
        print(f"DEBUG (gmail_search_action_node): Raw search results from service: '{search_results_from_service}'") # Log raw string

        if isinstance(search_results_from_service, str) and search_results_from_service.strip() != "":
            emails_data = search_results_from_service.strip().split("\n\n")
            for email_block in emails_data:
                lines = email_block.strip().split("\n")
                email_dict = {}
                for line in lines:
                    if ": " in line:
                        key, value = line.split(": ", 1)
                        key_normalized = key.strip().lower().replace(" ", "_")
                        email_dict[key_normalized] = value.strip()
                if email_dict.get("id"): 
                    parsed_email_summaries.append(email_dict)
            
            if not parsed_email_summaries and search_results_from_service.strip(): 
                 action_result_summary = "I found some email information, but had trouble summarizing it."
                 node_error_message = "Could not parse email summary string from MCP."
            elif not parsed_email_summaries: 
                 action_result_summary = f"I couldn't find any emails matching your search: '{params['query']}'."
        
        elif isinstance(search_results_from_service, list): 
            parsed_email_summaries = search_results_from_service # Assume already structured
            if not parsed_email_summaries:
                 action_result_summary = f"I couldn't find any emails matching your search: '{params['query']}'."

        elif isinstance(search_results_from_service, dict) and search_results_from_service.get("error"):
            error_detail = search_results_from_service.get("error")
            action_result_summary = f"Sorry, I couldn't search your emails. The service said: {error_detail}"
            node_error_message = action_result_summary
        
        # This handles the case where the service returns an empty string, meaning no results.
        elif isinstance(search_results_from_service, str) and search_results_from_service.strip() == "":
            action_result_summary = f"I couldn't find any emails matching your search: '{params['query']}'."
            parsed_email_summaries = [] # Ensure it's an empty list
            node_error_message = None # Not an error, just no results

        if parsed_email_summaries: # If we have successfully parsed summaries
            summary_parts = []
            for i, email_data in enumerate(parsed_email_summaries[:3]): 
                subject = email_data.get("subject", email_data.get("Subject", "No Subject")) 
                from_sender = email_data.get("from", email_data.get("From", "Unknown Sender"))
                summary_parts.append(f"{i+1}. Subject: '{subject}' (From: {from_sender})")
            if summary_parts:
                action_result_summary = f"I found {len(parsed_email_summaries)} email(s). Here are the first few:\n" + "\n".join(summary_parts)
            else:
                action_result_summary = f"I found {len(parsed_email_summaries)} email(s), but couldn't display a summary."
            node_error_message = None 
            
    print(f"DEBUG (gmail_search_action_node): Parsed email summaries for state: {parsed_email_summaries}")
    print(f"DEBUG (gmail_search_action_node): Action result summary for LLM: {action_result_summary}")
    return {**state, "tool_result": action_result_summary, "last_email_search_results": parsed_email_summaries, "error_message": node_error_message}


def gmail_read_action_node(state: AgentState) -> AgentState: # MODIFIED
    print("\n--- Node: Gmail Read Action ---")
    params = state.get("tool_parameters")
    action_result = "Could not understand which email to read."; node_error_message = state.get("error_message")
    
    if not gmail_service_instance: return {**state, "tool_result": "Gmail service unavailable.", "error_message": "Gmail service not initialized."}
    
    message_id_to_read = None
    if params and isinstance(params.get("message_id"), str):
        message_id_to_read = params["message_id"]
    # Positional logic was removed as LLM should resolve to message_id based on context
    
    if message_id_to_read:
        print(f"Attempting to read Gmail message ID: {message_id_to_read}")
        email_content_result = gmail_service_instance.invoke_tool(tool_name="read_email", tool_input={"messageId": message_id_to_read})
        
        if email_content_result and not (isinstance(email_content_result, dict) and email_content_result.get("error")):
            if isinstance(email_content_result, dict): # Expecting dict from service's parsing
                subject = email_content_result.get("subject", "No Subject")
                from_sender = email_content_result.get("from", "Unknown Sender")
                body_snippet = email_content_result.get("body", "No content available.")[:500] # Longer snippet
                action_result = f"Okay, here's the email from {from_sender} with subject '{subject}':\n\n{body_snippet}"
                if len(email_content_result.get("body", "")) > 500:
                    action_result += "\n\n(This is a snippet of a longer email.)"
            else: # Should be a dict if successful and parsed by service
                action_result = f"Received email content, but in an unexpected format: {str(email_content_result)[:300]}"
                node_error_message = "Unexpected format for read email content."
            node_error_message = None if not node_error_message else node_error_message # Preserve prior error if any
        else:
            error_detail = email_content_result.get("error") if isinstance(email_content_result, dict) else email_content_result
            action_result = f"Sorry, I couldn't read that email (ID: {message_id_to_read}). Gmail service said: {error_detail}"
            node_error_message = action_result
    elif not node_error_message: 
        node_error_message = "No message ID was determined to read the email."
        action_result = "I'm not sure which email you'd like me to read. Could you specify its ID or refer to it from a recent search?"

    return {**state, "tool_result": action_result, "error_message": node_error_message}


def generate_response_node(state: AgentState) -> AgentState: # MODIFIED
    print("\n--- Node: Generate Response ---")
    user_input = state["input"]; chat_history_messages = state.get("chat_history", [])
    user_profile_dict = state.get("user_profile", {}); retrieved_facts = state.get("retrieved_facts_for_query", [])
    tool_result_text = state.get("tool_result") 
    error_from_previous_node = state.get("error_message") # Error from param extraction or action node
    
    if not conversational_llm: 
        return {**state, "response": "LLM not available.", "error_message": "Conversational LLM not configured."}
    
    response_text = "I'm sorry, I had an issue processing that."; current_node_error = None
    try:
        user_facts_context_str = json.dumps(user_profile_dict) if user_profile_dict else "No specific user profile information available."
        
        retrieved_facts_context_str = "No specific facts were retrieved for this query."
        if retrieved_facts: 
            formatted_facts = []
            for fact_item in retrieved_facts: 
                fact_type = fact_item.get("type", fact_item.get("entity_type", "Fact"))
                details = fact_item.get("detail", fact_item.get("details", "N/A"))
                if isinstance(details, dict): details_str = ", ".join([f"{k}: {v}" for k,v in details.items()]); formatted_facts.append(f"- {fact_type}: {details_str}")
                elif isinstance(details, list): formatted_facts.append(f"- {fact_type}: {', '.join(map(str,details))}")
                else: formatted_facts.append(f"- {fact_type}: {details}")
            if formatted_facts: retrieved_facts_context_str = "Based on your question, I found these related facts:\n" + "\n".join(formatted_facts)
        
        # Construct context for the LLM
        # The ATHENA_SYSTEM_PROMPT (section 11) guides how to use 'Action result:'
        action_result_context = ""
        if tool_result_text:
            action_result_context = f"Action result: {tool_result_text}"
        
        # If there was an error from a previous node (like param extraction) and no tool_result yet,
        # make sure this error is highlighted for the LLM.
        system_note_for_llm = ""
        if error_from_previous_node:
            if tool_result_text and ("error" in str(tool_result_text).lower() or "sorry" in str(tool_result_text).lower()):
                # Error is already in tool_result, no need to duplicate
                pass
            else: # Error happened before tool_result or tool_result was not an error message itself
                system_note_for_llm = f"(System note: There was an issue: '{error_from_previous_node}'. Please acknowledge this gracefully if appropriate, then proceed with the user's original request: '{user_input}')"
        
        final_context_for_llm = f"{retrieved_facts_context_str}\n\n{action_result_context}".strip()
        current_input_for_llm = user_input
        if system_note_for_llm: # Prepend system note if it exists
            current_input_for_llm = f"{system_note_for_llm}\nUser: {user_input}"


        prompt_input_dict = {
            "user_facts_context": user_facts_context_str, 
            "retrieved_facts_context": final_context_for_llm, 
            "chat_history": chat_history_messages, 
            "input": current_input_for_llm 
        }
        print(f"Prompt input dict for conversational LLM: {json.dumps(prompt_input_dict, default=lambda o: '<BaseMessage object>' if isinstance(o, BaseMessage) else str(o), indent=2)}")
        prompt_messages_for_llm = ATHENA_SYSTEM_PROMPT.format_messages(**prompt_input_dict)
        
        llm_response_obj = conversational_llm.invoke(prompt_messages_for_llm)
        response_text = llm_response_obj.content.strip()
        if "<think>" in response_text: response_text = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()
    except Exception as e: 
        traceback.print_exc(); current_node_error = f"Error in generate_response: {e}"
        response_text = "I had a little trouble formulating that response. Could you try asking in a different way?" # More user-friendly general error
    
    print(f"Generated response: {response_text}")
    # Clear error_message if this node successfully generated a response that should have addressed it.
    # If this node itself had an error, current_node_error will be set.
    final_error_to_propagate = current_node_error
    if not current_node_error and error_from_previous_node:
        # If the LLM was prompted with the error and generated a response,
        # we can consider the error "handled" for this turn's state.
        # However, if the tool_result was an error, that's different.
        if not (tool_result_text and ("error" in str(tool_result_text).lower() or "sorry" in str(tool_result_text).lower())):
             final_error_to_propagate = None # Clear if error was not from tool and LLM responded

    return {**state, "response": response_text, "error_message": final_error_to_propagate}


# --- Graph Definition ---
workflow = StateGraph(AgentState)
workflow.add_node("decide_action", agent_decision_node)
workflow.add_node("extract_general_user_facts", extract_general_facts_node) 
workflow.add_node("extract_spotify_action_parameters", extract_spotify_action_parameters_node)
workflow.add_node("extract_gmail_send_parameters", extract_gmail_send_parameters_node)
workflow.add_node("extract_gmail_search_parameters", extract_gmail_search_parameters_node)
workflow.add_node("extract_gmail_read_parameters", extract_gmail_read_parameters_node)
workflow.add_node("query_user_facts", query_facts_node)
workflow.add_node("spotify_action", spotify_action_node) 
workflow.add_node("gmail_send_action", gmail_send_action_node)
workflow.add_node("gmail_search_action", gmail_search_action_node)
workflow.add_node("gmail_read_action", gmail_read_action_node)
workflow.add_node("generate_athena_response", generate_response_node)

workflow.set_entry_point("decide_action")

def route_after_decision(state: AgentState):
    decision = state.get("decision_outcome")
    if decision == "extract_facts": return "extract_general_user_facts"
    elif decision == "query_facts": return "query_user_facts"
    elif decision == "spotify_playback_action": return "extract_spotify_action_parameters" 
    elif decision == "gmail_send_email": return "extract_gmail_send_parameters"
    elif decision == "gmail_search_emails": return "extract_gmail_search_parameters"
    elif decision == "gmail_read_email": return "extract_gmail_read_parameters"
    elif decision == "exit": return END
    return "generate_athena_response" 

workflow.add_conditional_edges("decide_action", route_after_decision, {
    "extract_general_user_facts": "extract_general_user_facts",
    "extract_spotify_action_parameters": "extract_spotify_action_parameters", 
    "extract_gmail_send_parameters": "extract_gmail_send_parameters",
    "extract_gmail_search_parameters": "extract_gmail_search_parameters",
    "extract_gmail_read_parameters": "extract_gmail_read_parameters",
    "query_user_facts": "query_user_facts",
    "generate_athena_response": "generate_athena_response",
    END: END
})

workflow.add_edge("extract_general_user_facts", "generate_athena_response")
workflow.add_edge("query_user_facts", "generate_athena_response")
workflow.add_edge("extract_spotify_action_parameters", "spotify_action") 
workflow.add_edge("spotify_action", "generate_athena_response") 
workflow.add_edge("extract_gmail_send_parameters", "gmail_send_action")
workflow.add_edge("gmail_send_action", "generate_athena_response")
workflow.add_edge("extract_gmail_search_parameters", "gmail_search_action")
workflow.add_edge("gmail_search_action", "generate_athena_response")
workflow.add_edge("extract_gmail_read_parameters", "gmail_read_action")
workflow.add_edge("gmail_read_action", "generate_athena_response")
workflow.add_edge("generate_athena_response", END)

app_runnable = workflow.compile()

if __name__ == "__main__":
    print("Athena Agent - Conversational Loop Test with Corrected Spotify Routing & Artist Play")
    from app.core.database import Base, engine
    Base.metadata.create_all(bind=engine)
    db = next(get_db_session())
    main_test_user_username = "Abhay"
    test_user = crud_user.get_user_by_username(db, main_test_user_username)
    if not test_user:
        test_user = crud_user.create_user(db, username=main_test_user_username, location="Chatsville", hobbies=["conversing"])
    user_id = test_user.id
    user_profile_details = {
        "username": test_user.username, "location": test_user.location,
        "hobbies": test_user.hobbies_json, "jobs": test_user.job_json,
        "preferences": test_user.preferences_json
    }
    db.close()
    print(f"Starting conversation with User ID: {user_id}, Username: {user_profile_details['username']}")
    current_chat_history: List[BaseMessage] = []
    while True:
        user_input_text = input(f"{user_profile_details['username']}: ").strip()
        if not user_input_text: continue
        if user_input_text.lower() == "exit":
            print("Athena: It was lovely chatting with you. Goodbye!"); break
        
        history_for_agent = current_chat_history.copy()
        current_turn_state = AgentState(
            input=user_input_text, chat_history=history_for_agent, user_id=user_id,
            user_profile=user_profile_details, decision_outcome=None, tool_parameters=None, tool_result=None,
            extracted_entities_json=None, retrieved_facts_for_query=None, response=None, error_message=None
        )
        print("\nInvoking agent graph...")
        try:
            final_state = app_runnable.invoke(current_turn_state, config={}) 
            athena_response = final_state.get("response", "I'm not sure how to respond to that.")
            if final_state.get("decision_outcome") == "exit" and not final_state.get("response"): 
                athena_response = "It was lovely chatting with you. Goodbye for now! 👋"
            if not isinstance(athena_response, str): 
                athena_response = "I had a slight issue with my response, please try again."

            print(f"Athena: {athena_response}")
            current_chat_history.append(HumanMessage(content=user_input_text))
            current_chat_history.append(AIMessage(content=athena_response))
            
            if final_state.get("error_message"):
                print(f"Debug - Error from last turn: {final_state['error_message']}")
        except Exception as e:
            print(f"CRITICAL ERROR during graph invocation: {e}"); traceback.print_exc()
            print("Athena: I'm having some trouble right now. Let's try again in a moment.")
    print("\nConversation ended.")

