from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func # For server-side default timestamps
from ..core.database import Base # Import Base from your database configuration

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    username = Column(String, unique=True, index=True, nullable=False)

    # User-specific top-level attributes that are relatively stable
    location = Column(String, nullable=True)
    hobbies_json = Column(JSON, nullable=True, comment="Stores a list of strings for user's hobbies")
    job_json = Column(JSON, nullable=True, comment="Stores a list of strings for user's job titles")
    preferences_json = Column(JSON, nullable=True, comment="Stores a list of strings for user's preferences")

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())

    # Relationships
    # All dynamic entities related to the user will be stored in UserEntity
    entities = relationship("UserEntity", back_populates="user", cascade="all, delete-orphan")
    chat_sessions = relationship("ChatSession", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}')>"

class UserEntity(Base):
    __tablename__ = "user_entities"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # This column stores the TYPE of entity as determined by the LLM.
    # e.g., "pet", "family_member", "event", "car", "book_series", "reminder", "project", "goal", "medication_schedule"
    entity_type = Column(String, index=True, nullable=False, comment="Type of entity (e.g., pet, event, car, etc.)")
                                                
    # All details for this specific entity instance go into this JSON blob.
    # The structure within this JSON will be guided by your FACT_EXTRACTION_TEMPLATE.
    # Example for entity_type="pet": {"name": "Max", "species": "dog", "breed": "Golden Retriever"}
    # Example for entity_type="event": {"title": "Doctor's Appointment", "date_original_text": "next Tuesday at 3pm", "interpreted_datetime": "YYYY-MM-DDTHH:MM:SS"}
    # Example for entity_type="car": {"make": "Toyota", "model": "Camry", "year": 2020, "color": "Blue"}
    details_json = Column(JSON, nullable=False, comment="All attributes of the entity instance")

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())

    user = relationship("User", back_populates="entities")

    def __repr__(self):
        return f"<UserEntity(id={self.id}, user_id={self.user_id}, entity_type='{self.entity_type}')>"

class ChatSession(Base):
    __tablename__ = "chat_sessions"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    session_uuid = Column(String, unique=True, index=True, nullable=False)
    start_time = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="chat_sessions")
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("chat_sessions.id"), nullable=False)
    role = Column(String, nullable=False)  # "user", "assistant", "system"
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    
    session = relationship("ChatSession", back_populates="messages")

if __name__ == "__main__":
    from app.core.database import engine
    print("Dropping existing tables (if any) and creating new ones with UserEntity model...")
    Base.metadata.drop_all(bind=engine) 
    Base.metadata.create_all(bind=engine)
    print("Database tables created/recreated with UserEntity model.")