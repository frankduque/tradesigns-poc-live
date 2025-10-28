"""
Database connection and session management
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.ext.declarative import declarative_base
from src.config import DATABASE_URL

# Engine
engine = create_engine(
    DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,  # Verifica conexes antes de usar
    echo=False  # Set True para debug SQL
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Session = scoped_session(SessionLocal)

# Base para models
Base = declarative_base()

def get_db():
    """Dependency para obter sesso do banco"""
    db = Session()
    try:
        yield db
    finally:
        db.close()

def get_db_sync():
    """Retorna sesso sncrona (para uso direto)"""
    return Session()

def close_db():
    """Fecha todas as conexes"""
    Session.remove()
    engine.dispose()
