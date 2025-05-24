from sqlalchemy.orm import Session
from ..models import persistent_models # Your SQLAlchemy models (User, UserEntity)
from ..core.database import SessionLocal # To get a DB session
import json # For working with JSON details

# --- UserEntity CRUD Operations ---

def create_user_entity(db: Session, user_id: int, entity_type: str, details: dict) -> persistent_models.UserEntity:
    """
    Creates a new entity associated with a user.
    'details' is a Python dictionary that will be stored as JSON.
    """
    if not isinstance(details, dict):
        raise ValueError("Entity details must be a dictionary.")

    db_entity = persistent_models.UserEntity(
        user_id=user_id,
        entity_type=entity_type,
        details_json=details  # SQLAlchemy handles the dict to JSON conversion
    )
    db.add(db_entity)
    db.commit()
    db.refresh(db_entity)
    print(f"Entity type '{entity_type}' created for user ID {user_id} with entity ID {db_entity.id}.")
    return db_entity

def get_user_entity_by_id(db: Session, entity_id: int) -> persistent_models.UserEntity | None:
    """
    Retrieves a specific entity by its ID.
    """
    return db.query(persistent_models.UserEntity).filter(persistent_models.UserEntity.id == entity_id).first()

def get_user_entities(db: Session, user_id: int, entity_type: str | None = None, limit: int = 100, skip: int = 0) -> list[persistent_models.UserEntity]:
    """
    Retrieves all entities for a given user, optionally filtered by entity_type.
    Includes pagination.
    """
    query = db.query(persistent_models.UserEntity).filter(persistent_models.UserEntity.user_id == user_id)
    if entity_type:
        query = query.filter(persistent_models.UserEntity.entity_type == entity_type)
    return query.offset(skip).limit(limit).all()

def update_user_entity_details(db: Session, entity_id: int, new_details: dict) -> persistent_models.UserEntity | None:
    """
    Updates the details_json of an existing entity.
    'new_details' is a Python dictionary. This function will overwrite the existing details_json.
    For merging/patching, more complex logic would be needed here or in a service layer.
    """
    db_entity = get_user_entity_by_id(db, entity_id)
    if db_entity:
        if not isinstance(new_details, dict):
            raise ValueError("New details must be a dictionary.")
        db_entity.details_json = new_details # Overwrites existing JSON
        db.commit()
        db.refresh(db_entity)
        print(f"Entity ID {entity_id} details updated successfully.")
        return db_entity
    print(f"Entity ID {entity_id} not found for update.")
    return None

def merge_user_entity_details(db: Session, entity_id: int, details_to_merge: dict) -> persistent_models.UserEntity | None:
    """
    Merges new details into the existing details_json of an entity.
    'details_to_merge' is a Python dictionary.
    This performs a shallow merge (top-level keys from details_to_merge will overwrite).
    For deep merging, a more sophisticated merge utility would be needed.
    """
    db_entity = get_user_entity_by_id(db, entity_id)
    if db_entity:
        if not isinstance(details_to_merge, dict):
            raise ValueError("Details to merge must be a dictionary.")
        
        current_details = db_entity.details_json
        if not isinstance(current_details, dict): # Should always be a dict if created properly
            current_details = {}
            
        merged_details = current_details.copy() # Start with current details
        merged_details.update(details_to_merge) # Merge/overwrite with new details
        
        db_entity.details_json = merged_details
        db.commit()
        db.refresh(db_entity)
        print(f"Entity ID {entity_id} details merged successfully.")
        return db_entity
    print(f"Entity ID {entity_id} not found for merging details.")
    return None


def delete_user_entity(db: Session, entity_id: int) -> bool:
    """
    Deletes an entity by its ID.
    Returns True if deletion was successful, False otherwise.
    """
    db_entity = get_user_entity_by_id(db, entity_id)
    if db_entity:
        db.delete(db_entity)
        db.commit()
        print(f"Entity ID {entity_id} deleted successfully.")
        return True
    print(f"Entity ID {entity_id} not found for deletion.")
    return False

# --- Example Usage (for testing this file directly) ---
if __name__ == "__main__":
    from app.core.database import Base, engine
    from app.crud.crud_user import create_user, get_user_by_username, delete_user # For test setup

    # Ensure tables exist
    Base.metadata.create_all(bind=engine)

    db: Session = SessionLocal()

    print("Testing UserEntity CRUD operations...")

    # 1. Create a test user first (or get an existing one)
    test_username_for_entity = "entity_testuser"
    db_user = get_user_by_username(db, test_username_for_entity)
    if not db_user:
        db_user = create_user(db, username=test_username_for_entity, location="EntityTestLand")
    
    if not db_user:
        print("Failed to create or retrieve a test user. Aborting UserEntity CRUD tests.")
        db.close()
        exit()
    
    user_id_for_entity_test = db_user.id
    print(f"Using User ID: {user_id_for_entity_test} for entity tests.")

    # 2. Test create_user_entity
    print("\n--- Testing create_user_entity ---")
    pet_details = {"name": "Buddy", "species": "dog", "breed": "Golden Retriever", "loves": ["walks", "treats"]}
    event_details = {"description": "Team Meeting", "date_text": "Tomorrow 10 AM", "location": "Virtual"}
    
    created_pet_entity = create_user_entity(db, user_id=user_id_for_entity_test, entity_type="pet", details=pet_details)
    created_event_entity = create_user_entity(db, user_id=user_id_for_entity_test, entity_type="event", details=event_details)

    if created_pet_entity:
        print(f"Created pet entity: ID {created_pet_entity.id}, Type: {created_pet_entity.entity_type}, Details: {created_pet_entity.details_json}")
        entity_id_for_test = created_pet_entity.id

        # 3. Test get_user_entity_by_id
        print("\n--- Testing get_user_entity_by_id ---")
        retrieved_entity = get_user_entity_by_id(db, entity_id_for_test)
        if retrieved_entity:
            print(f"Retrieved entity: ID {retrieved_entity.id}, Details: {retrieved_entity.details_json}")
            assert retrieved_entity.details_json["name"] == "Buddy"
        else:
            print(f"Failed to retrieve entity ID {entity_id_for_test}")

        # 4. Test get_user_entities
        print("\n--- Testing get_user_entities (all for user) ---")
        all_entities = get_user_entities(db, user_id=user_id_for_entity_test)
        print(f"Found {len(all_entities)} entities for user ID {user_id_for_entity_test}.")
        for entity in all_entities:
            print(f"  - ID: {entity.id}, Type: {entity.entity_type}, Name (if any): {entity.details_json.get('name', entity.details_json.get('description'))}")
        assert len(all_entities) >= 2 # We created two

        print("\n--- Testing get_user_entities (filtered by type 'pet') ---")
        pet_entities = get_user_entities(db, user_id=user_id_for_entity_test, entity_type="pet")
        print(f"Found {len(pet_entities)} 'pet' entities.")
        assert len(pet_entities) >= 1
        if pet_entities:
             print(f"  First pet name: {pet_entities[0].details_json.get('name')}")


        # 5. Test update_user_entity_details (overwrite)
        print("\n--- Testing update_user_entity_details (overwrite) ---")
        updated_pet_details_overwrite = {"name": "Buddy II", "species": "dog", "breed": "Labrador", "mood": "happy"}
        updated_entity = update_user_entity_details(db, entity_id_for_test, updated_pet_details_overwrite)
        if updated_entity:
            print(f"Updated entity (overwrite): ID {updated_entity.id}, Details: {updated_entity.details_json}")
            assert updated_entity.details_json["name"] == "Buddy II"
            assert "loves" not in updated_entity.details_json # 'loves' field should be gone

        # 6. Test merge_user_entity_details
        print("\n--- Testing merge_user_entity_details ---")
        details_to_merge_in = {"mood": "playful", "favorite_toy": "squeaky ball", "breed": "Super Lab"} # breed will overwrite
        merged_entity = merge_user_entity_details(db, entity_id_for_test, details_to_merge_in)
        if merged_entity:
            print(f"Merged entity: ID {merged_entity.id}, Details: {merged_entity.details_json}")
            assert merged_entity.details_json["name"] == "Buddy II" # Name should persist
            assert merged_entity.details_json["mood"] == "playful" # Mood updated
            assert merged_entity.details_json["breed"] == "Super Lab" # Breed overwritten
            assert merged_entity.details_json["favorite_toy"] == "squeaky ball" # New field added
        
        # 7. Test delete_user_entity
        # print("\n--- Testing delete_user_entity ---")
        # if created_event_entity and delete_user_entity(db, created_event_entity.id):
        #     deleted_check = get_user_entity_by_id(db, created_event_entity.id)
        #     if not deleted_check:
        #         print(f"Event entity ID {created_event_entity.id} confirmed deleted.")
        #     else:
        #         print(f"Error: Event entity ID {created_event_entity.id} still found after delete call.")
        # else:
        #     print(f"Delete entity call failed or event entity was not created for ID {created_event_entity.id if created_event_entity else 'N/A'}")
            
    # Clean up the test user (optional, keeps DB clean for next run)
    # print(f"\nCleaning up test user '{test_username_for_entity}'...")
    # delete_user(db, user_id_for_entity_test)

    db.close()
    print("\nUserEntity CRUD tests finished.")
