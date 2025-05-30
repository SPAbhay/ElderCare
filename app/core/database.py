# app/core/database.py

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

# Load environment variables from .env file at the project root
# Assuming database.py is in app/core/, .env is two levels up.
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)
    print("DEBUG (database.py): Loaded .env file for DATABASE_URL.")
else:
    print("DEBUG (database.py): .env file not found at expected location, relying on system environment variables for DATABASE_URL.")

# Get the database URL from environment variable
# For local development, this will come from your .env file.
# For Streamlit Cloud, you'll set this in the app's secrets.
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    print("CRITICAL ERROR: DATABASE_URL environment variable not set.")
    # Fallback to local SQLite if cloud DB URL isn't set, for local testing convenience
    # However, for deployment, DATABASE_URL *must* be set to the cloud DB.
    print("Warning: DATABASE_URL not set. Falling back to local SQLite 'athena_chat.db' for this session.")
    print("Ensure DATABASE_URL is set in your .env file or Streamlit secrets for proper operation.")
    DATABASE_URL = "sqlite:///./athena_chat.db" 
    # If falling back to SQLite, add connect_args again
    engine = create_engine(
        DATABASE_URL, connect_args={"check_same_thread": False}
    )
else:
    print(f"DEBUG (database.py): Connecting to database: {DATABASE_URL.split('@')[-1]}") # Avoid printing credentials
    # For PostgreSQL (and most other non-SQLite DBs), remove connect_args={"check_same_thread": False}
    engine = create_engine(DATABASE_URL)


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db_session_for_context_manager(): # Renamed to avoid conflict if you have other get_db
    """Dependency to get a DB session, for use with 'with' statement."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Optional: Function to create all tables (useful for initial setup)
# This is typically called by your main app (streamlit_app.py or test scripts)
def init_db():
    print("Initializing database and creating tables if they don't exist...")
    try:
        # Import all modules here that define models so that
        # Base has them registered before create_all is called
        from app.models import persistent_models # This will ensure User, UserEntity are known to Base
        Base.metadata.create_all(bind=engine)
        print("Database tables checked/created.")
    except Exception as e:
        print(f"Error during table creation: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    print(f"Attempting to initialize database using URL from environment (or fallback).")
    print(f"Actual DATABASE_URL being used for this test: {'PostgreSQL (Cloud)' if 'postgresql' in DATABASE_URL else 'SQLite (Local Fallback)'}")
    init_db()
    print("Database initialization process finished.")
    # You can add test queries here if needed
    # Example:
    # db_gen = get_db_session_for_context_manager()
    # db: Session = next(db_gen)
    # try:
    #     # Try a simple query if it's PostgreSQL
    #     if "postgresql" in DATABASE_URL:
    #         result = db.execute(text("SELECT 1")).scalar_one()
    #         print(f"Test query to PostgreSQL successful, result: {result}")
    #     else:
    #         print("Skipping direct DB test query for SQLite fallback in this test.")
    # except Exception as e:
    #     print(f"Test query failed: {e}")
    # finally:
    #     db.close()

