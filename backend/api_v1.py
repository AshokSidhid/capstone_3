import torch
from fastapi import FastAPI, HTTPException, status, Depends
from pydantic import BaseModel, Field
from typing import List, Dict, Union, Any
from sqlalchemy.orm import Session

# Database & Model Imports
from backend.database import init_db, get_db
from backend.models_db import TelegramPostDB, ModelPredictionDB
from models.joint_model import JointManipulationSpanModel
from backend.calibration import TemperatureScaler, ConfidenceCalibrationEngine

# Initialize FastAPI App
app = FastAPI(
    title="LANG-71 Ukrainian Manipulation Observatory API",
    version="1.0.0",
    description="Joint Multilingual Transformer Inference Service with Calibrated Abstention & SQLite Storage"
)

# Auto-create SQLite tables on startup
@app.on_event("startup")
def startup_event():
    init_db()

# --- Schemas ---

class PostAnalysisRequest(BaseModel):
    post_id: str = Field(..., example="tg_post_10482")
    channel_id: str = Field(..., example="channel_ukr_news")
    text: str = Field(..., example="Це очевидна маніпуляція та ворожа пропаганда!")

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
    # Flexible mapping to accept either plain floats OR dicts if needed
    class_probabilities: Dict[str, Union[float, Dict[str, Any]]]


# --- Model and Calibration Engine Setup ---
LABELS = [
    "Loaded_Language", "Name_Calling", "Repetition", "Exaggeration",
    "Doubt", "Appeal_to_Authority", "Flag_Waving", 
    "Causal_Oversimplification", "False_Dilemma", "Straw_Man"
]

model = JointManipulationSpanModel(num_labels=len(LABELS))
model.eval()
scaler = TemperatureScaler(temperature=1.35)
calibration_engine = ConfidenceCalibrationEngine(scaler=scaler, threshold=0.65)


# --- API Endpoint ---

@app.post("/api/v1/analyze", response_model=PostAnalysisResponse)
async def analyze_telegram_post(payload: PostAnalysisRequest, db: Session = Depends(get_db)):
    try:
        # 1. Forward Pass with Dummy Tensors
        dummy_input = torch.randint(0, 1000, (1, 128))
        dummy_mask = torch.ones((1, 128), dtype=torch.long)
        
        with torch.no_grad():
            cls_logits, span_logits = model(dummy_input, dummy_mask)
        
        # 2. Calibration & Abstention Evaluation
        calibration_result = calibration_engine.evaluate_with_abstention(
            raw_logits=cls_logits, 
            label_names=LABELS
        )
        
        # 3. Span Extraction Logic
        extracted_spans = [
            EvidenceSpan(text_segment=payload.text[0:25], start_char=0, end_char=25)
        ] if calibration_result["detected_techniques"] else []

        # 4. Save to Local SQLite Database
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

        # 5. Return Validated Response
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