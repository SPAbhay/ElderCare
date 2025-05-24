import streamlit as st
from sqlalchemy.orm import Session
import traceback

# Project imports
from app.core.database import SessionLocal, Base, engine
from app.crud import crud_user
from app.agent.agent_graph import app_runnable, AgentState
from langchain_core.messages import HumanMessage, AIMessage

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        
st.set_page_config(page_title="Athena - ElderCare Companion", layout="wide")
st.title("👵🏽 Athena - Your Caring Companion")
st.caption("Powered by LangGraph and Advanced AI Models")

# Initialize session state variables if they don't exist
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "username" not in st.session_state:
    st.session_state.username = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [] # Will store BaseMessage objects
if "agent_initialized" not in st.session_state:
    st.session_state.agent_initialized = False # To track if user profile is loaded for agent

# --- User Identification / Login ---
if not st.session_state.user_id:
    st.subheader("Welcome! Please tell me your name to begin.")
    username_input = st.text_input("Enter your name:", key="username_login")

    if st.button("Start Chatting", key="login_button"):
        if username_input:
            db_gen = get_db()
            db: Session = next(db_gen)
            try:
                user = crud_user.get_user_by_username(db, username=username_input)
                if not user:
                    # For simplicity in MVP, let's pre-fill some details or leave them for later extraction
                    user = crud_user.create_user(
                        db, 
                        username=username_input,
                        location="Not specified yet", # Example default
                        hobbies=[],
                        jobs=[],
                        preferences=[]
                    )
                    st.toast(f"Welcome, {username_input}! Your new profile has been created.", icon="🎉")
                else:
                    st.toast(f"Welcome back, {username_input}!", icon="👋")

                st.session_state.user_id = user.id
                st.session_state.username = user.username
                st.session_state.agent_initialized = True # Mark as ready for agent

                # Load existing entities or user profile summary for the agent state if needed
                # For now, agent_graph.py's nodes will fetch fresh user profile data as needed.
                # We might load chat history from DB later.

                st.rerun() # Rerun the script to move past the login section
            finally:
                db.close()
        else:
            st.error("Please enter your name.")
else:
    # --- Main App Interface after Login ---
    st.sidebar.success(f"Logged in as: {st.session_state.username} (ID: {st.session_state.user_id})")

    # Clear chat history button
    if st.sidebar.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.toast("Chat history cleared!", icon="🧹")
        st.rerun()

    st.sidebar.button("Logout", on_click=lambda: (
        st.session_state.clear(),
        st.rerun()
    ))

    st.header(f"Chat with Athena 💬")

    # Display existing chat messages
    for message in st.session_state.chat_history:
        with st.chat_message(message.type): # message.type will be "human" or "ai"
            st.markdown(message.content)

    # Chat input
    if prompt := st.chat_input("What would you like to talk about?"):
        # Add user message to chat history
        st.session_state.chat_history.append(HumanMessage(content=prompt))

        # Display user message in chat message container
        with st.chat_message("human"):
            st.markdown(prompt)

        # --- Call Athena Agent ---
        with st.chat_message("ai"):
            with st.spinner("Athena is thinking..."):
                try:
                    # Prepare the state for the agent
                    # Ensure user_profile is loaded if not already in session_state or passed directly
                    if "user_profile_details" not in st.session_state:
                        # Fetch user profile details if not already available
                        # This ensures the agent has the latest profile for its context
                        db = next(get_db())
                        try:
                            user_obj = crud_user.get_user(db, st.session_state.user_id)
                            if user_obj:
                                st.session_state.user_profile_details = {
                                    "username": user_obj.username,
                                    "location": user_obj.location,
                                    "hobbies": user_obj.hobbies_json,
                                    "jobs": user_obj.job_json,
                                    "preferences": user_obj.preferences_json
                                }
                            else: # Should not happen if user is logged in
                                st.session_state.user_profile_details = {"username": st.session_state.username}
                        finally:
                            db.close()

                    current_turn_state = AgentState(
                        input=prompt,
                        chat_history=st.session_state.chat_history.copy(), # Pass the history up to this point
                        user_id=st.session_state.user_id,
                        user_profile=st.session_state.user_profile_details,
                        # Other fields will be populated by the agent
                        decision_outcome=None,
                        extracted_entities_json=None,
                        retrieved_facts_for_query=None,
                        response=None,
                        error_message=None
                    )

                    # Invoke the agent (LangGraph runnable)
                    # The config is empty here because we are not using persistent checkpoints in this basic loop
                    # For multi-user or persistent memory across app restarts, you'd use checkpointer with a config.
                    final_state = app_runnable.invoke(current_turn_state, config={}) 
                    athena_response_content = final_state.get("response", "I'm not sure how to respond to that right now.")
                    
                    if final_state.get("decision_outcome") == "exit":
                        athena_response_content = "It was lovely chatting with you. Goodbye for now! 👋"
                        st.markdown(athena_response_content)
                        st.session_state.chat_history.append(AIMessage(content=athena_response_content))
                        # Further actions for exit can be added here (e.g., disable input)
                    else:
                        # Proceed with normal response handling
                        potential_response = final_state.get("response")
                        # ... (rest of the logic to ensure athena_response_content is a string) ...
                        st.markdown(athena_response_content)
                        st.session_state.chat_history.append(AIMessage(content=athena_response_content))

                    if final_state.get("error_message"):
                        st.error(f"Athena encountered an issue: {final_state['error_message']}")
                        # Optionally, you can choose to display a generic error or the detailed one
                        # athena_response_content = "I had a little trouble with that request."

                except Exception as e:
                    st.error(f"An error occurred while communicating with Athena: {e}")
                    athena_response_content = "I'm having some trouble connecting right now. Please try again in a moment."
                    # Log the full error for debugging
                    print(f"CRITICAL ERROR in Streamlit during agent call: {e}")
                    traceback.print_exc()


                st.markdown(athena_response_content)

        # Add AI's actual response to chat history
        st.session_state.chat_history.append(AIMessage(content=athena_response_content))

        # Rerun to display the new messages immediately (optional, Streamlit often handles this)
        st.rerun() 