import datetime
from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, Text, ForeignKey, JSON
from sqlalchemy.orm import relationship
from backend.database import Base

class TelegramPostDB(Base):
    """Stores ingested Telegram posts along with channel and split metadata."""
    __tablename__ = "telegram_posts"

    post_id = Column(String, primary_key=True, index=True)
    channel_id = Column(String, nullable=False, index=True)
    text = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    partition_split = Column(String, default="inference")

    # Relationships
    prediction = relationship("ModelPredictionDB", back_populates="post", uselist=False)
    adjudication = relationship("AnalystAdjudicationDB", back_populates="post", uselist=False)


class ModelPredictionDB(Base):
    """Stores joint XLM-RoBERTa predictions, calibrated probabilities, and extracted evidence spans."""
    __tablename__ = "model_predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    post_id = Column(String, ForeignKey("telegram_posts.post_id"), nullable=False, unique=True)
    
    status = Column(String, nullable=False)  # "VERIFIED_AUTOMATED" or "ABSTAIN_FLAGGED_FOR_REVIEW"
    max_confidence = Column(Float, nullable=False)
    requires_human_adjudication = Column(Boolean, default=False)
    
    # JSON columns for storing array/dictionary structures in SQLite
    detected_techniques = Column(JSON, nullable=False)  # e.g., ["Loaded_Language"]
    extracted_spans = Column(JSON, nullable=False)        # e.g., [{"text": "...", "start": 0, "end": 20}]
    class_probabilities = Column(JSON, nullable=False)    # Raw & calibrated probability dict
    
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationships
    post = relationship("TelegramPostDB", back_populates="prediction")


class AnalystAdjudicationDB(Base):
    """Stores human-in-the-loop analyst feedback, manual overrides, and audit trails."""
    __tablename__ = "analyst_adjudications"

    id = Column(Integer, primary_key=True, autoincrement=True)
    post_id = Column(String, ForeignKey("telegram_posts.post_id"), nullable=False, unique=True)
    
    analyst_id = Column(String, nullable=False)
    decision = Column(String, nullable=False)  # "APPROVED", "OVERRIDDEN", or "REJECTED"
    final_techniques = Column(JSON, nullable=False)
    analyst_notes = Column(Text, nullable=True)
    
    reviewed_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationships
    post = relationship("TelegramPostDB", back_populates="adjudication")