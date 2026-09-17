import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Local SQLite Database Path
DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
os.makedirs(DB_DIR, exist_ok=True)
SQLITE_DB_PATH = os.path.join(DB_DIR, "observatory.db")

# SQLite Database URL
SQLALCHEMY_DATABASE_URL = f"sqlite:///{SQLITE_DB_PATH}"

# Connect args for SQLite to support multi-threaded FastAPI execution
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, 
    connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    """FastAPI Dependency for Database Session Management."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """Utility function to auto-create SQLite tables on application startup."""
    import backend.models_db  # Import models to register metadata
    Base.metadata.create_all(bind=engine)