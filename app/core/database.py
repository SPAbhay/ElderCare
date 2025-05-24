from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

DATABASE_URL = "sqlite:///./athena_chat.db"

# Create the SQLAlchemy engine
# connect_args is needed for SQLite to handle multithreading correctly with FastAPI
engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)

# Create a SessionLocal class, which will be used to create database sessions
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create a Base class for our declarative models
# Our database models will inherit from this class
Base = declarative_base()

# Optional: Function to create all tables (useful for initial setup)
def create_db_and_tables():
    # This function is called to create the tables defined in your models
    # We'll import models here when they are defined to avoid circular imports
    # For now, it's a placeholder for where table creation logic will go.
    # Example: from app.models import user_model # Assuming user_model.py contains User table
    # Base.metadata.create_all(bind=engine)
    print("Database tables would be created here if models were imported and defined.")

if __name__ == "__main__":
    # This part is for standalone testing/setup if you run this file directly
    print(f"Attempting to initialize database at: {os.path.abspath('athena_chat.db')}")
    # We will call create_db_and_tables() from main.py or a dedicated script later,
    # once models are defined.
    # For now, this just confirms the path.