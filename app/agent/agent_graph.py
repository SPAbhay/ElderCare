# app/agent/agent_graph.py

from typing import TypedDict, Annotated, List, Dict, Any, Optional
import operator
import json
import traceback # For detailed error logging
import re # For more robustly finding JSON
import os # For os.getenv

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage 
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver 

from sqlalchemy.orm import Session
from sqlalchemy import text 
from pydantic import BaseModel, ValidationError 

# Project-specific imports
from app.core.llm_config import get_llm 
from app.prompts.prompt_templates import ROUTING_PROMPT, GENERIC_ENTITY_EXTRACTION_PROMPT, ATHENA_SYSTEM_PROMPT
from app.crud import crud_user, crud_entity 
from app.core.database import SessionLocal 
from app.utils.temporal_parser import interpret_temporal_term 

# --- Agent State Definition ---
class AgentState(TypedDict):
    input: str                              
    chat_history: Annotated[List[BaseMessage], operator.add] 
    user_id: int                            
    user_profile: Optional[dict]            
    decision_outcome: Optional[str]         
    extracted_entities_json: Optional[list] # List of dicts: {"id": ..., "entity_type": ..., "details_json": ...}
    retrieved_facts_for_query: Optional[List[dict]] 
    response: Optional[str]                 
    error_message: Optional[str]            

# --- Pydantic model for Query Structure Validation ---
class ParsedQuery(BaseModel):
    query_entity_type: Optional[str] = None
    query_identifier: Optional[str] = None
    query_attributes: Optional[List[str]] = None

# --- LLM Initialization ---
# User-specified model to be used as the default for all tasks,
# unless overridden by specific environment variables.
USER_SPECIFIED_DEFAULT_MODEL = "qwen/qwen3-235b-a22b:free" # As per your .env

ROUTING_MODEL_NAME = os.getenv("ROUTING_MODEL_NAME", "meta-llama/llama-3.3-70b-instruct:free")
FACT_EXTRACTION_MODEL_NAME = os.getenv("FACT_EXTRACTION_MODEL_NAME", "meta-llama/llama-3.3-70b-instruct:free")
QUERY_UNDERSTANDING_MODEL_NAME = os.getenv("QUERY_UNDERSTANDING_MODEL_NAME", USER_SPECIFIED_DEFAULT_MODEL) # qwen/qwen3-235b-a22b:free
CONVERSATIONAL_MODEL_NAME = os.getenv("CONVERSATIONAL_MODEL_NAME", "meta-llama/llama-3.3-70b-instruct:free") 

print(f"Using ROUTING_MODEL_NAME: {ROUTING_MODEL_NAME}")
print(f"Using FACT_EXTRACTION_MODEL_NAME: {FACT_EXTRACTION_MODEL_NAME}")
print(f"Using QUERY_UNDERSTANDING_MODEL_NAME: {QUERY_UNDERSTANDING_MODEL_NAME}")
print(f"Using CONVERSATIONAL_MODEL_NAME: {CONVERSATIONAL_MODEL_NAME}")

fact_extraction_llm = get_llm(model_name=FACT_EXTRACTION_MODEL_NAME, temperature=0.1) 
routing_llm = get_llm(model_name=ROUTING_MODEL_NAME, temperature=0.0) 
query_understanding_llm = get_llm(model_name=QUERY_UNDERSTANDING_MODEL_NAME, temperature=0.0) 
conversational_llm = get_llm(model_name=CONVERSATIONAL_MODEL_NAME, temperature=0.7)

# --- Node Definitions ---

def get_db_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def agent_decision_node(state: AgentState) -> AgentState:
    print("\n--- Node: Agent Decision ---")
    user_input = state["input"]
    user_id = state["user_id"]
    node_error_message = None
    decision = "generate_response" 
    print(f"Routing input: '{user_input}' for user_id: {user_id}")

    if not routing_llm:
        print("ERROR: Routing LLM not available.")
        return {**state, "decision_outcome": decision, "error_message": "Routing LLM not configured."}
    try:
        db = next(get_db_session())
        user_facts_for_prompt_str = "{}" 
        try:
            current_user = crud_user.get_user(db, user_id)
            if current_user:
                user_profile_summary = {
                    "username": current_user.username, "location": current_user.location,
                    "hobbies": current_user.hobbies_json, "jobs": current_user.job_json,
                    "preferences": current_user.preferences_json
                }
                user_facts_for_prompt_str = json.dumps(user_profile_summary)
        finally:
            db.close()
        
        prompt_messages = ROUTING_PROMPT.format_messages(user_facts=user_facts_for_prompt_str, input=user_input)
        print(f"Sending prompt to LLM ({routing_llm.model_name if hasattr(routing_llm, 'model_name') else 'N/A'}) for routing decision...")
        llm_response = routing_llm.invoke(prompt_messages)
        raw_llm_output_for_routing = llm_response.content.strip()
        print(f"LLM raw output for routing: '{raw_llm_output_for_routing}'")

        extracted_decision_keyword = raw_llm_output_for_routing
        if "</think>" in extracted_decision_keyword: # Handle potential <think> tags
            extracted_decision_keyword = extracted_decision_keyword.split("</think>")[-1].strip()
        
        # Take only the first word as the keyword
        extracted_decision_keyword = extracted_decision_keyword.split(maxsplit=1)[0].lower() if extracted_decision_keyword else ""
        print(f"Cleaned decision keyword: '{extracted_decision_keyword}'")

        allowed_decisions = ["extract_facts", "query_facts", "generate_response", "exit", "planning_query", "other"]
        if extracted_decision_keyword in allowed_decisions: decision = extracted_decision_keyword
        else: 
            print(f"Warning: Routing LLM returned an unexpected decision keyword: '{extracted_decision_keyword}' (from raw: '{raw_llm_output_for_routing}'). Defaulting to 'generate_response'.")
            node_error_message = f"Routing LLM returned unexpected decision: {extracted_decision_keyword}"
            
    except Exception as e:
        print(f"ERROR: An unexpected error occurred in agent_decision_node: {e}")
        traceback.print_exc(); node_error_message = f"Core error in agent_decision_node: {e}"
    print(f"Final Decision: {decision}")
    # Clear previous query results if not querying again
    return {
        **state, 
        "decision_outcome": decision, 
        "retrieved_facts_for_query": [] if decision != "query_facts" else state.get("retrieved_facts_for_query"), # Clear if not querying
        "error_message": node_error_message if node_error_message else state.get("error_message")
    }


def extract_facts_node(state: AgentState) -> AgentState:
    print("\n--- Node: Extract Facts ---")
    user_input = state["input"]; user_id = state["user_id"]
    processed_entities_for_state = []; node_error_message = None
    print(f"Attempting to extract facts from input: '{user_input}' for user_id: {user_id}")
    if not fact_extraction_llm: return {**state, "error_message": "Fact LLM not configured.", "extracted_entities_json": []}
    try:
        prompt_messages = GENERIC_ENTITY_EXTRACTION_PROMPT.format_messages(input=user_input)
        print(f"Sending prompt to LLM ({fact_extraction_llm.model_name if hasattr(fact_extraction_llm, 'model_name') else 'N/A'}) for fact extraction...")
        llm_response = fact_extraction_llm.invoke(prompt_messages)
        llm_output_json_str = llm_response.content
        print(f"LLM raw output for fact extraction:\n<<<\n{llm_output_json_str}\n>>>")
        
        potential_json = None; parsed_output = None; identified_entities = []
        try:
            stripped_llm_output = llm_output_json_str
            if "<think>" in stripped_llm_output: 
                 stripped_llm_output = re.sub(r"<think>.*?</think>", "", stripped_llm_output, flags=re.DOTALL).strip()
            
            match = re.search(r"```json\s*(\{.*?\})\s*```", stripped_llm_output, re.DOTALL)
            if match: potential_json = match.group(1).strip()
            else: 
                match = re.search(r"(\{.*?\})", stripped_llm_output, re.DOTALL) 
                if match: potential_json = match.group(1).strip()

            if potential_json:
                print(f"Attempting to parse extracted JSON block:\n<<<\n{potential_json}\n>>>")
                decoder = json.JSONDecoder(); parsed_output, end_index = decoder.raw_decode(potential_json)
                remaining_text = potential_json[end_index:].strip()
                if remaining_text and remaining_text != "": 
                    print(f"WARNING: Additional data found after the first JSON object for fact extraction: '{remaining_text[:100]}...'")

                if parsed_output and "identified_entities" in parsed_output:
                    identified_entities = parsed_output.get("identified_entities", [])
                    if not isinstance(identified_entities, list):
                        node_error_message = "LLM output for entities was not a list."
                        identified_entities = []
                elif parsed_output: node_error_message = "Parsed JSON missing 'identified_entities' key."
                else: node_error_message = "Parsed_output is None after raw_decode, unexpected."
            else: node_error_message = "No JSON object pattern found in LLM output for fact extraction."
        except json.JSONDecodeError as e_parse: 
            node_error_message = f"Parse error for fact extraction: {e_parse}"
            print(f"Problematic JSON for fact extraction: {potential_json if potential_json else llm_output_json_str}")
        except Exception as e_parse_other:
            node_error_message = f"Unexpected parse error for fact extraction: {e_parse_other}"; traceback.print_exc()

        if identified_entities:
            print(f"LLM identified {len(identified_entities)} entities. Processing them...")
            db = next(get_db_session())
            try:
                for entity_data in identified_entities:
                    if isinstance(entity_data, dict) and "entity_type" in entity_data and "details" in entity_data:
                        entity_type = entity_data["entity_type"]
                        details_json = entity_data["details"]
                        if entity_type == "event" and isinstance(details_json, dict) and "date_text" in details_json and isinstance(details_json["date_text"], str) :
                            temporal_interpretation = interpret_temporal_term(details_json["date_text"])
                            details_json["temporal_interpretation"] = temporal_interpretation
                        created_entity = crud_entity.create_user_entity(db, user_id, entity_type, details_json)
                        if created_entity: 
                            processed_entities_for_state.append({"id": created_entity.id, "entity_type": created_entity.entity_type, "details_json": created_entity.details_json})
            finally: db.close()
        elif not node_error_message: print("LLM identified no new entities to extract.")
    except Exception as e: node_error_message = f"Core error in extract_facts_node: {e}"; traceback.print_exc()
    return {**state, "extracted_entities_json": processed_entities_for_state, "error_message": node_error_message or state.get("error_message")}


def query_facts_node(state: AgentState) -> AgentState:
    print("\n--- Node: Query Facts ---")
    user_input = state["input"]; user_id = state["user_id"]
    retrieved_facts_for_response = []; node_error_message = None
    print(f"Attempting to understand query: '{user_input}' for user_id: {user_id}")

    if not query_understanding_llm:
        return {**state, "retrieved_facts_for_query": [], "error_message": "Query LLM not configured."}
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
        common_entity_types = "personal_info, user_hobby, user_job, user_preference_general, pet, family_member, event, reminder_shopping, vehicle_maintenance, etc."
        formatted_parser_prompt = query_parser_prompt_template.format(question=user_input, entity_type_examples=common_entity_types)
        print(f"Sending prompt to LLM ({query_understanding_llm.model_name if hasattr(query_understanding_llm, 'model_name') else 'N/A'}) for query understanding...")
        llm_response = query_understanding_llm.invoke(formatted_parser_prompt) 
        raw_query_structure_str = llm_response.content if hasattr(llm_response, 'content') else llm_response
        print(f"LLM raw output for query understanding:\n<<<\n{raw_query_structure_str}\n>>>")

        parsed_query_dict: Optional[Dict[str, Any]] = None 
        validated_query_structure: Optional[ParsedQuery] = None 
        potential_query_json_str = None 
        try:
            stripped_llm_output = raw_query_structure_str
            if "<think>" in stripped_llm_output: 
                 stripped_llm_output = re.sub(r"<think>.*?</think>", "", stripped_llm_output, flags=re.DOTALL).strip()
            match = re.search(r"```json\s*(\{.*?\})\s*```", stripped_llm_output, re.DOTALL)
            if match: potential_query_json_str = match.group(1).strip()
            else: 
                match = re.search(r"(\{.*?\})", stripped_llm_output, re.DOTALL) 
                if match: potential_query_json_str = match.group(1).strip()
            
            if potential_query_json_str:
                print(f"Attempting to parse extracted JSON block for query structure:\n<<<\n{potential_query_json_str}\n>>>")
                decoder = json.JSONDecoder()
                parsed_query_dict, end_index = decoder.raw_decode(potential_query_json_str)
                remaining_text = potential_query_json_str[end_index:].strip()
                if remaining_text and remaining_text != "": 
                    print(f"WARNING: Additional data found after first JSON in query structure: '{remaining_text[:100]}...'")
                try:
                    validated_query_structure = ParsedQuery.model_validate(parsed_query_dict)
                    print(f"Validated query structure (Pydantic): {validated_query_structure.model_dump()}")
                except ValidationError as e_pydantic:
                    node_error_message = f"Query structure Pydantic validation failed: {e_pydantic}"
            else: node_error_message = "No JSON object pattern found in query LLM output."
        except json.JSONDecodeError as e_json:
            node_error_message = f"Query structure JSON parse error: {e_json}"
            print(f"Problematic query JSON string: {potential_query_json_str or raw_query_structure_str}")
        except Exception as e_parse: 
            traceback.print_exc(); node_error_message = f"Unexpected query parse error: {e_parse}"

        if validated_query_structure: 
            query_entity_type = validated_query_structure.query_entity_type
            query_identifier = validated_query_structure.query_identifier
            query_attributes = validated_query_structure.query_attributes
            db = next(get_db_session())
            try:
                if query_entity_type in ["personal_info", "user_location", "user_hobby", "user_job", "user_preference_general"]:
                    user_profile_db = crud_user.get_user(db, user_id) 
                    if user_profile_db:
                        if query_entity_type in ["personal_info", "user_location"] and (not query_attributes or "location" in query_attributes) and user_profile_db.location:
                            retrieved_facts_for_response.append({"type": "location", "detail": user_profile_db.location})
                        if query_entity_type == "user_hobby" and (not query_attributes) and user_profile_db.hobbies_json: # If no specific attrs, return all
                            retrieved_facts_for_response.append({"type": "hobbies", "details": user_profile_db.hobbies_json})
                        if query_entity_type == "user_job" and (not query_attributes) and user_profile_db.job_json:
                             retrieved_facts_for_response.append({"type": "jobs", "details": user_profile_db.job_json})
                        # Add user_preference_general if needed
                elif query_entity_type: 
                    entities_from_db = crud_entity.get_user_entities(db, user_id=user_id, entity_type=query_entity_type)
                    print(f"Found {len(entities_from_db)} entities of type '{query_entity_type}' for user {user_id}.")
                    for entity in entities_from_db:
                        details = entity.details_json
                        if not isinstance(details, dict): continue
                        entity_matches_identifier = True 
                        if query_identifier:
                            id_lower = str(query_identifier).lower()
                            name_match = str(details.get("name","")).lower() == id_lower
                            desc_match = str(details.get("description","")).lower() == id_lower
                            title_match = str(details.get("title","")).lower() == id_lower
                            date_text_match = str(details.get("date_text","")).lower().find(id_lower) != -1 if query_entity_type == "event" else False
                            entity_matches_identifier = name_match or desc_match or title_match or date_text_match
                        if entity_matches_identifier:
                            fact_detail_to_add = {"entity_type": entity.entity_type}
                            if query_attributes and isinstance(query_attributes, list) and len(query_attributes) > 0: 
                                specific_details = {}
                                for attr in query_attributes:
                                    if attr in details: specific_details[attr] = details[attr]
                                if specific_details: fact_detail_to_add["details"] = specific_details
                                else: continue 
                            else: fact_detail_to_add["details"] = details
                            if "details" in fact_detail_to_add and fact_detail_to_add["details"]: 
                                 retrieved_facts_for_response.append(fact_detail_to_add)
            finally: db.close()
        if not retrieved_facts_for_response and not node_error_message:
            print("No specific facts found for the query based on parsed structure.")
    except Exception as e:
        traceback.print_exc(); node_error_message = f"Core error in query_facts_node: {e}"
    print(f"Retrieved facts for query: {json.dumps(retrieved_facts_for_response, indent=2)}")
    return {**state, "retrieved_facts_for_query": retrieved_facts_for_response, "error_message": node_error_message or state.get("error_message")}


def generate_response_node(state: AgentState) -> AgentState:
    print("\n--- Node: Generate Response ---")
    user_input = state["input"]
    chat_history_messages = state.get("chat_history", [])
    user_profile_dict = state.get("user_profile", {}) 
    retrieved_facts = state.get("retrieved_facts_for_query", [])
    node_error_message = state.get("error_message") # Carry over errors from previous nodes
    
    print(f"Generating response for input: '{user_input}'")
    print(f"Chat history length: {len(chat_history_messages)}")
    print(f"User profile for context: {user_profile_dict}")
    print(f"Retrieved facts for context: {retrieved_facts}")
    if node_error_message:
        print(f"Error from previous node: {node_error_message}")

    if not conversational_llm:
        print("ERROR: Conversational LLM not available.")
        return {**state, "response": "I'm having a little trouble thinking right now. Please try again later.", "error_message": "Conversational LLM not configured."}

    response_text = "I'm sorry, I encountered an issue and couldn't fully process that." # Default error response

    try:
        user_facts_context_str = json.dumps(user_profile_dict) if user_profile_dict else "No specific user profile information available."
        retrieved_facts_context_str = "No specific facts were retrieved for this query."
        if retrieved_facts:
            formatted_facts = []
            for fact_item in retrieved_facts: # Renamed fact to fact_item
                fact_type = fact_item.get("type", fact_item.get("entity_type", "Fact"))
                details = fact_item.get("detail", fact_item.get("details", "N/A"))
                if isinstance(details, dict):
                    details_str = ", ".join([f"{k}: {v}" for k,v in details.items()])
                    formatted_facts.append(f"- {fact_type}: {details_str}")
                elif isinstance(details, list):
                    formatted_facts.append(f"- {fact_type}: {', '.join(map(str,details))}")
                else:
                    formatted_facts.append(f"- {fact_type}: {details}")
            if formatted_facts:
                retrieved_facts_context_str = "Based on your question, I found these related facts:\n" + "\n".join(formatted_facts)
            else: # Handles case where retrieved_facts might be [{}] or similar non-informative structures
                retrieved_facts_context_str = "I looked up some information but didn't find specific details to share for that query."
        
        # Construct the prompt input dictionary
        prompt_input_dict = {
            "user_facts_context": user_facts_context_str,
            "retrieved_facts_context": retrieved_facts_context_str,
            "chat_history": chat_history_messages, 
            "input": user_input 
        }
        
        # Add error context if an error occurred in a previous node
        if node_error_message:
            # Prepend a message to the input or add to context, so LLM is aware of the issue
            # This is a simple way; more sophisticated error handling could involve specific error prompts
            prompt_input_dict["input"] = f"(System note: There was an issue: {node_error_message}. Please respond to the user appropriately regarding their original input: '{user_input}')"
            print(f"Input to conversational LLM modified due to error: {prompt_input_dict['input']}")


        print(f"Prompt input dict for conversational LLM: {json.dumps(prompt_input_dict, default=lambda o: '<BaseMessage object>' if isinstance(o, BaseMessage) else str(o), indent=2)}")

        prompt_messages_for_llm = ATHENA_SYSTEM_PROMPT.format_messages(**prompt_input_dict)
        
        print(f"Sending prompt to Conversational LLM ({conversational_llm.model_name if hasattr(conversational_llm, 'model_name') else 'N/A'})...")
        llm_response_obj = conversational_llm.invoke(prompt_messages_for_llm)
        response_text = llm_response_obj.content.strip()
        
        if "<think>" in response_text: # Clean <think> tags
            response_text = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()
        
        print(f"Generated response: {response_text}")

    except Exception as e:
        print(f"ERROR: An unexpected error occurred in generate_response_node: {e}")
        traceback.print_exc()
        node_error_message = f"Core error in generate_response_node: {e}"
        # response_text remains the default error response set above

    return {
        **state, 
        "response": response_text,
        # Clear error message after attempting to generate a response for it, 
        # or carry it if generation itself failed.
        "error_message": node_error_message if "Core error in generate_response_node" in (node_error_message or "") else None 
    }


# --- Graph Definition ---
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("decide_action", agent_decision_node)
workflow.add_node("extract_user_facts", extract_facts_node)
workflow.add_node("query_user_facts", query_facts_node)
workflow.add_node("generate_athena_response", generate_response_node)

# Set entry point
workflow.set_entry_point("decide_action")

# Define edges
def route_after_decision(state: AgentState):
    decision = state.get("decision_outcome")
    if decision == "extract_facts":
        return "extract_user_facts"
    elif decision == "query_facts":
        return "query_user_facts"
    elif decision == "generate_response" or decision == "other" or decision == "planning_query": # Route these to generate_response
        return "generate_athena_response"
    elif decision == "exit":
        return END
    return "generate_athena_response" # Default fallback

workflow.add_conditional_edges(
    "decide_action",
    route_after_decision,
    {
        "extract_user_facts": "extract_user_facts",
        "query_user_facts": "query_user_facts",
        "generate_athena_response": "generate_athena_response",
        END: END
    }
)

# After extracting or querying facts, always generate a response
workflow.add_edge("extract_user_facts", "generate_athena_response")
workflow.add_edge("query_user_facts", "generate_athena_response")

# After generating a response, the current turn processing ends.
# The main loop will handle if the conversation continues.
workflow.add_edge("generate_athena_response", END)


# Compile the graph
# For persisting history with SQLiteSaver:
# memory = SqliteSaver.from_conn_string(":memory:") # Or a file path like "athena_chat_memory.db"
# app_runnable = workflow.compile(checkpointer=memory)

# For now, compile without checkpointer for simpler loop management
app_runnable = workflow.compile()


if __name__ == "__main__":
    print("Athena Agent - Conversational Loop Test")
    from app.core.database import Base, engine
    Base.metadata.create_all(bind=engine)
    
    # --- User Setup ---
    db = next(get_db_session())
    main_test_user_username = "AthenaTester"
    test_user = crud_user.get_user_by_username(db, main_test_user_username)
    if not test_user:
        test_user = crud_user.create_user(db, username=main_test_user_username, location="Chatsville", hobbies=["conversing"])
    
    # Store user details for passing to AgentState
    user_id = test_user.id
    user_profile_details = {
        "username": test_user.username,
        "location": test_user.location,
        "hobbies": test_user.hobbies_json,
        "jobs": test_user.job_json,
        "preferences": test_user.preferences_json
    }
    db.close()
    print(f"Starting conversation with User ID: {user_id}, Username: {user_profile_details['username']}")

    # --- Conversational Loop ---
    current_chat_history: List[BaseMessage] = []

    while True:
        user_input_text = input(f"{user_profile_details['username']}: ").strip()
        if not user_input_text:
            continue
        if user_input_text.lower() == "exit":
            print("Athena: It was lovely chatting with you. Goodbye!")
            break

        # Prepare state for the graph
        current_turn_state = AgentState(
            input=user_input_text,
            chat_history=current_chat_history.copy(), # Pass a copy of the current history
            user_id=user_id,
            user_profile=user_profile_details,
            decision_outcome=None,
            extracted_entities_json=None,
            retrieved_facts_for_query=None,
            response=None,
            error_message=None
        )

        print("\nInvoking agent graph...")
        try:
            # For graphs compiled without checkpointer, config is usually empty or for thread_id
            # If using checkpointer, config would be e.g. {"configurable": {"thread_id": str(user_id)}}
            final_state = app_runnable.invoke(current_turn_state, config={}) 
            
            athena_response = final_state.get("response", "I'm not sure how to respond to that.")
            print(f"Athena: {athena_response}")

            # Update chat history for the next turn
            current_chat_history.append(HumanMessage(content=user_input_text))
            current_chat_history.append(AIMessage(content=athena_response))

            if final_state.get("error_message"):
                print(f"Debug - Error from last turn: {final_state['error_message']}")
            
            # Optional: Save chat messages to DB if desired (not part of AgentState persistence here)
            # This would require creating ChatSession and ChatMessage records.

        except Exception as e:
            print(f"CRITICAL ERROR during graph invocation: {e}")
            traceback.print_exc()
            print("Athena: I'm having some trouble right now. Let's try again in a moment.")
            # Potentially break or offer to restart

    print("\nConversation ended.")

