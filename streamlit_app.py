# streamlit_app.py
import streamlit as st
from sqlalchemy.orm import Session
import json 
import traceback 
import os # For environment variables in Streamlit Cloud

# Project imports
from app.core.database import SessionLocal, Base, engine 
from app.crud import crud_user 
from app.agent.agent_graph import app_runnable, AgentState 
from langchain_core.messages import HumanMessage, AIMessage 

# Attempt to load environment variables if a .env file exists (for local testing)
# On Streamlit Cloud, secrets will be set in the dashboard.
from dotenv import load_dotenv
if os.path.exists(".env"):
    load_dotenv()
    print("Loaded .env file for local development via streamlit_app.py")


# Ensure database tables are created (idempotent operation)
try:
    # This import ensures SQLAlchemy models are registered with Base
    from app.models import persistent_models 
    Base.metadata.create_all(bind=engine)
    print("Database tables checked/created via Streamlit app.")
except ImportError:
    print("Warning: Could not import persistent_models. Ensure app.models.persistent_models exists.")
except Exception as e:
    print(f"Error during table creation in Streamlit: {e}")
    traceback.print_exc()


# Helper to get a DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Streamlit App ---

st.set_page_config(page_title="Athena - ElderCare Companion", layout="wide", initial_sidebar_state="expanded")
st.title("👵🏽 Athena - Your Caring Companion")
st.caption("Powered by LangGraph and Advanced AI Models")

# Initialize session state variables if they don't exist
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "username" not in st.session_state:
    st.session_state.username = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [] 
if "user_profile_details" not in st.session_state: 
    st.session_state.user_profile_details = None


# --- User Identification / Login ---
if not st.session_state.user_id:
    st.subheader("Welcome! Please tell me your name to begin or continue.")
    with st.form(key="login_form_streamlit"): # Ensure form key is unique if other forms exist
        username_input = st.text_input("Enter your name:", key="username_login_input_streamlit_app") 
        login_button = st.form_submit_button("Start Chatting")
        
        if login_button:
            if username_input:
                db_gen = get_db()
                db: Session = next(db_gen)
                try:
                    user = crud_user.get_user_by_username(db, username=username_input)
                    if not user:
                        user = crud_user.create_user(
                            db, 
                            username=username_input,
                            location="Not specified yet", 
                            hobbies=[], jobs=[], preferences=[]
                        )
                        st.toast(f"Welcome, {username_input}! Your new profile has been created.", icon="�")
                    else:
                        st.toast(f"Welcome back, {username_input}!", icon="👋")
                    
                    st.session_state.user_id = user.id
                    st.session_state.username = user.username
                    st.session_state.user_profile_details = {
                        "username": user.username, "location": user.location,
                        "hobbies": user.hobbies_json, "jobs": user.job_json,
                        "preferences": user.preferences_json
                    }
                    st.session_state.chat_history = [] 
                    st.rerun() 
                except Exception as e_login:
                    st.error(f"Error during login/user creation: {e_login}")
                    traceback.print_exc()
                finally:
                    db.close()
            else:
                st.error("Please enter your name.")
else:
    # --- Main App Interface after Login ---
    st.sidebar.success(f"Logged in as: {st.session_state.username} (ID: {st.session_state.user_id})")
    
    if st.sidebar.button("Clear Chat History", key="clear_chat_streamlit_main"):
        st.session_state.chat_history = []
        st.toast("Chat history cleared!", icon="🧹")
        st.rerun()

    def logout():
        keys_to_clear = list(st.session_state.keys())
        for key in keys_to_clear:
            del st.session_state[key]
        st.rerun()

    st.sidebar.button("Logout", on_click=logout, key="logout_streamlit_main")

    st.header(f"Chat with Athena 💬")

    # Display existing chat messages
    for message_obj in st.session_state.chat_history: 
        if hasattr(message_obj, 'type') and hasattr(message_obj, 'content') and isinstance(message_obj.content, str):
            with st.chat_message(message_obj.type): # Key removed
                st.markdown(message_obj.content)
        else: 
             with st.chat_message("error"): # Key removed
                st.warning(f"Malformed message in history: {type(message_obj)}")


    if prompt := st.chat_input("What would you like to talk about?"):
        st.session_state.chat_history.append(HumanMessage(content=prompt))
        # User message will be displayed on the next rerun due to the loop above

        # --- MODIFICATION: Removed key from this st.chat_message call ---
        with st.chat_message("ai"): 
            with st.spinner("Athena is thinking..."):
                athena_response_content = "I'm having a little trouble connecting right now. Please try again in a moment." 
                try:
                    if not st.session_state.user_profile_details or \
                       st.session_state.user_profile_details.get("username") != st.session_state.username:
                        db = next(get_db())
                        try:
                            user_obj = crud_user.get_user(db, st.session_state.user_id)
                            if user_obj:
                                st.session_state.user_profile_details = {
                                    "username": user_obj.username, "location": user_obj.location,
                                    "hobbies": user_obj.hobbies_json, "jobs": user_obj.job_json,
                                    "preferences": user_obj.preferences_json
                                }
                            else: 
                                st.session_state.user_profile_details = {"username": st.session_state.username} 
                        finally:
                            db.close()
                    
                    history_for_agent = st.session_state.chat_history[:-1] 

                    current_turn_state = AgentState(
                        input=prompt,
                        chat_history=history_for_agent, 
                        user_id=st.session_state.user_id,
                        user_profile=st.session_state.user_profile_details,
                        decision_outcome=None, extracted_entities_json=None,
                        tool_parameters=None, tool_result=None, 
                        retrieved_facts_for_query=None, response=None, error_message=None,
                        last_email_search_results=st.session_state.get("last_email_search_results") 
                    )

                    print(f"\nInvoking agent graph from Streamlit with input: {current_turn_state['input']}")
                    final_state = app_runnable.invoke(current_turn_state, config={}) 
                    print(f"Final state from agent: {final_state}")
                    
                    if "last_email_search_results" in final_state:
                        st.session_state.last_email_search_results = final_state.get("last_email_search_results")

                    if final_state.get("decision_outcome") == "exit":
                        athena_response_content = "It was lovely chatting with you. Goodbye for now! 👋"
                    else:
                        potential_response = final_state.get("response")
                        if isinstance(potential_response, str):
                            athena_response_content = potential_response
                        elif potential_response is None and not final_state.get("error_message"):
                            athena_response_content = "I'm not quite sure how to respond to that. Could you try rephrasing?"
                        elif final_state.get("error_message"):
                            athena_response_content = potential_response if isinstance(potential_response, str) else "I encountered an issue. Please try again."
                        else: 
                            athena_response_content = "I seem to be having a little trouble forming a response. Please try again."
                        
                except Exception as e:
                    st.error(f"An error occurred while communicating with Athena: {str(e)[:500]}")
                    athena_response_content = "I'm having some trouble connecting right now. Please try again in a moment."
                    print(f"CRITICAL ERROR in Streamlit during agent call: {e}")
                    traceback.print_exc()
                
                st.markdown(athena_response_content) 
        
        if not isinstance(athena_response_content, str):
            athena_response_content = "Error: Response was not a string." 
        st.session_state.chat_history.append(AIMessage(content=athena_response_content))
        
        st.rerun() 