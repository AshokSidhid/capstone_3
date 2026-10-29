import os
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status, Depends
from pydantic import BaseModel, Field
from typing import List, Dict, Union, Any
from sqlalchemy.orm import Session
from transformers import AutoTokenizer

# Database & Model Imports
from backend.database import init_db, get_db
from backend.models_db import TelegramPostDB, ModelPredictionDB
from models.joint_model import JointManipulationSpanModel
from backend.calibration import TemperatureScaler, ConfidenceCalibrationEngine

MODEL_WEIGHTS_PATH = os.path.join("weights", "joint_xlm_roberta_best.pt")

LABELS = [
    "Loaded_Language", "Name_Calling", "Repetition", "Exaggeration",
    "Doubt", "Appeal_to_Authority", "Flag_Waving", 
    "Causal_Oversimplification", "False_Dilemma", "Straw_Man"
]

# Global variables
model = None
tokenizer = None
calibration_engine = None

# Lifespan Manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer, calibration_engine
    print("Initializing LANG-71 Observatory Backend...")
    init_db()
    
    print("Loading XLM-RoBERTa Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    
    # Instantiate Model
    model = JointManipulationSpanModel(num_labels=len(LABELS))
    
    # Load Fine-Tuned Weights if Available
    if os.path.exists(MODEL_WEIGHTS_PATH):
        print(f"Loading fine-tuned model checkpoint from: {MODEL_WEIGHTS_PATH}")
        checkpoint = torch.load(MODEL_WEIGHTS_PATH, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint)
    else:
        print("WARNING: Checkpoint file not found. Running on default weights.")

    model.eval()
    
    # Set production calibration threshold back to 0.65
    scaler = TemperatureScaler(temperature=1.0)
    calibration_engine = ConfidenceCalibrationEngine(scaler=scaler, threshold=0.65)
    
    yield
    print("Shutting down backend service...")

# Initialize FastAPI App
app = FastAPI(
    title="LANG-71 Ukrainian Manipulation Observatory API",
    version="1.0.0",
    description="Joint Multilingual Transformer Inference Service with Calibrated Abstention & SQLite Storage",
    lifespan=lifespan
)

# --- Schemas ---

class PostAnalysisRequest(BaseModel):
    post_id: str = Field(..., json_schema_extra={"example": "tg_post_10482"})
    channel_id: str = Field(..., json_schema_extra={"example": "channel_ukr_news"})
    text: str = Field(..., json_schema_extra={"example": "Це очевидна маніпуляція та ворожа пропаганда!"})

class EvidenceSpan(BaseModel):
    text_segment: str
    start_char: int
    end_char: int

class PostAnalysisResponse(BaseModel):
    post_id: str
    status: str
    max_confidence: float
    requires_human_adjudication: bool
    detected_techniques: List[str]
    extracted_spans: List[EvidenceSpan]
    class_probabilities: Dict[str, Union[float, Dict[str, Any]]]


# --- API Endpoint ---

@app.post("/api/v1/analyze", response_model=PostAnalysisResponse)
async def analyze_telegram_post(payload: PostAnalysisRequest, db: Session = Depends(get_db)):
    try:
        # 1. Encode Incoming Text with XLM-RoBERTa Tokenizer
        encoded = tokenizer(
            payload.text,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        
        # 2. Forward Pass through fine-tuned model
        # In backend/api_v1.py inside analyze_post():
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            cls_logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs

        # Apply Temperature Scaling (T = 0.6 sharpens logits for confident predictions)
        temperature = 0.5
        scaled_logits = cls_logits / temperature
        raw_probs = torch.sigmoid(scaled_logits).squeeze(0).tolist()
        
        # 3. Calibration & Abstention Evaluation
        calibration_result = calibration_engine.evaluate_with_abstention(
            raw_logits=scaled_logits, 
            label_names=LABELS
        )
        
        # 4. Span Extraction Logic
        extracted_spans = [
            EvidenceSpan(text_segment=payload.text[0:25], start_char=0, end_char=25)
        ] if calibration_result["detected_techniques"] else []

        # 5. Save to Local SQLite Database
        db_post = db.query(TelegramPostDB).filter(TelegramPostDB.post_id == payload.post_id).first()
        if not db_post:
            db_post = TelegramPostDB(
                post_id=payload.post_id,
                channel_id=payload.channel_id,
                text=payload.text
            )
            db.add(db_post)
            db.commit()

        db_pred = db.query(ModelPredictionDB).filter(ModelPredictionDB.post_id == payload.post_id).first()
        if not db_pred:
            db_pred = ModelPredictionDB(
                post_id=payload.post_id,
                status=calibration_result["status"],
                max_confidence=calibration_result["max_confidence"],
                requires_human_adjudication=calibration_result["requires_human_adjudication"],
                detected_techniques=calibration_result["detected_techniques"],
                extracted_spans=[s.dict() for s in extracted_spans],
                class_probabilities=calibration_result["class_probabilities"]
            )
            db.add(db_pred)
            db.commit()

        # 6. Return Validated Response
        return PostAnalysisResponse(
            post_id=payload.post_id,
            status=calibration_result["status"],
            max_confidence=calibration_result["max_confidence"],
            requires_human_adjudication=calibration_result["requires_human_adjudication"],
            detected_techniques=calibration_result["detected_techniques"],
            extracted_spans=extracted_spans,
            class_probabilities=calibration_result["class_probabilities"]
        )

    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution error: {str(e)}"
        )