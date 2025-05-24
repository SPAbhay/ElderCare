from sqlalchemy.orm import Session
from ..models import persistent_models # Your SQLAlchemy models
from ..core.database import SessionLocal # To get a DB session

# --- User CRUD Operations ---

def get_user(db: Session, user_id: int) -> persistent_models.User | None:
    """
    Retrieves a user by their ID.
    """
    return db.query(persistent_models.User).filter(persistent_models.User.id == user_id).first()

def get_user_by_username(db: Session, username: str) -> persistent_models.User | None:
    """
    Retrieves a user by their username.
    """
    return db.query(persistent_models.User).filter(persistent_models.User.username == username).first()

def create_user(db: Session, username: str, location: str | None = None, 
                hobbies: list[str] | None = None, jobs: list[str] | None = None, 
                preferences: list[str] | None = None) -> persistent_models.User:
    """
    Creates a new user.
    Handles JSON fields for hobbies, jobs, and preferences.
    """
    db_user = persistent_models.User(
        username=username,
        location=location,
        hobbies_json=hobbies if hobbies else [],
        job_json=jobs if jobs else [],
        preferences_json=preferences if preferences else []
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    print(f"User '{username}' created successfully with ID {db_user.id}.")
    return db_user

def update_user(db: Session, user_id: int, updates: dict) -> persistent_models.User | None:
    """
    Updates an existing user's information.
    'updates' is a dictionary of fields to update.
    Handles JSON fields correctly.
    """
    db_user = get_user(db, user_id)
    if db_user:
        for key, value in updates.items():
            if hasattr(db_user, key):
                # Special handling for JSON list fields if needed,
                # e.g., to append or overwrite. For now, direct setattr.
                if key in ["hobbies_json", "job_json", "preferences_json"] and value is None:
                    setattr(db_user, key, []) # Ensure it's an empty list if None is passed
                else:
                    setattr(db_user, key, value)
            else:
                print(f"Warning: User model does not have attribute '{key}'. Skipping update for this field.")

        db.commit()
        db.refresh(db_user)
        print(f"User ID {user_id} updated successfully.")
        return db_user
    print(f"User ID {user_id} not found for update.")
    return None

def delete_user(db: Session, user_id: int) -> bool:
    """
    Deletes a user by their ID.
    Returns True if deletion was successful, False otherwise.
    """
    db_user = get_user(db, user_id)
    if db_user:
        db.delete(db_user)
        db.commit()
        print(f"User ID {user_id} deleted successfully.")
        return True
    print(f"User ID {user_id} not found for deletion.")
    return False

# --- Example Usage (for testing this file directly) ---
if __name__ == "__main__":
    # This is a simple way to test CRUD operations.
    # In a real app, SessionLocal would be managed by FastAPI dependencies.

    # Create tables if they don't exist (ensure models are loaded by SQLAlchemy's Base)
    from app.core.database import Base, engine
    Base.metadata.create_all(bind=engine) # Ensures tables exist for the test

    db: Session = SessionLocal()

    print("Testing User CRUD operations...")

    # Test create_user
    print("\n--- Testing create_user ---")
    test_username = "testuser_crud"
    # Clean up if user already exists from a previous test run
    existing_user_for_test = get_user_by_username(db, test_username)
    if existing_user_for_test:
        print(f"User '{test_username}' already exists. Deleting for fresh test.")
        delete_user(db, existing_user_for_test.id)

    created_user = create_user(
        db, 
        username=test_username, 
        location="Testville",
        hobbies=["reading", "coding"],
        jobs=["developer"],
        preferences=["coffee"]
    )
    if created_user:
        print(f"Created user: {created_user.username}, Location: {created_user.location}, Hobbies: {created_user.hobbies_json}")
        user_id_for_test = created_user.id

        # Test get_user
        print("\n--- Testing get_user ---")
        retrieved_user = get_user(db, user_id_for_test)
        if retrieved_user:
            print(f"Retrieved user: {retrieved_user.username}, Hobbies: {retrieved_user.hobbies_json}")
        else:
            print(f"Failed to retrieve user ID {user_id_for_test}")

        # Test get_user_by_username
        print("\n--- Testing get_user_by_username ---")
        retrieved_by_username = get_user_by_username(db, test_username)
        if retrieved_by_username:
            print(f"Retrieved by username: {retrieved_by_username.username}")
        else:
            print(f"Failed to retrieve user '{test_username}' by username.")

        # Test update_user
        print("\n--- Testing update_user ---")
        updates_to_apply = {
            "location": "UpdatedCity",
            "hobbies_json": ["reading", "hiking", "photography"], # Overwrite hobbies
            "preferences_json": None # Test setting a JSON field to empty list
        }
        updated_user = update_user(db, user_id_for_test, updates_to_apply)
        if updated_user:
            print(f"Updated user: Location: {updated_user.location}, Hobbies: {updated_user.hobbies_json}, Preferences: {updated_user.preferences_json}")

        retrieved_after_update = get_user(db, user_id_for_test)
        if retrieved_after_update:
             print(f"State after update: Location: {retrieved_after_update.location}, Hobbies: {retrieved_after_update.hobbies_json}, Prefs: {retrieved_after_update.preferences_json}")


        # Test delete_user
        # print("\n--- Testing delete_user ---")
        # if delete_user(db, user_id_for_test):
        #     deleted_check = get_user(db, user_id_for_test)
        #     if not deleted_check:
        #         print(f"User ID {user_id_for_test} confirmed deleted.")
        #     else:
        #         print(f"Error: User ID {user_id_for_test} still found after delete call.")
        # else:
        #      print(f"Delete user call returned false for user ID {user_id_for_test}")

    else:
        print(f"Failed to create user '{test_username}' for testing.")

    db.close()
    print("\nUser CRUD tests finished.")