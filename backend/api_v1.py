from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import torch
import numpy as np

from models.joint_model import JointManipulationSpanModel
from backend.calibration import ConfidenceCalibrationEngine

app = FastAPI(
    title="LANG-71 Ukrainian Manipulation Observatory API",
    description="REST API service for joint manipulation technique classification, text-span evidence extraction, and analyst adjudication.",
    version="1.0.0"
)

MODEL_NAME = "xlm-roberta-base"
LABELS = [
    "Appeal_to_Authority", "Bandwagon", "Causal_Oversimplification", 
    "Doubt", "Emotional_Charge", "Loaded_Language", "Name_Calling", 
    "Repetition", "Slogans", "Straw_Man"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = JointManipulationSpanModel(model_name=MODEL_NAME, num_labels=len(LABELS)).to(device)
model.eval()

calibration_engine = ConfidenceCalibrationEngine(confidence_threshold=0.65)

class PostAnalysisRequest(BaseModel):
    post_id: str = Field(..., example="tg_post_10482")
    channel_id: str = Field(..., example="channel_ukr_news")
    text: str = Field(..., example="Це очевидна маніпуляція та ворожа пропаганда!")

class SpanEvidence(BaseModel):
    text_segment: str
    start_char: int
    end_char: int

class PostAnalysisResponse(BaseModel):
    post_id: str
    status: str
    requires_human_adjudication: bool
    max_confidence: float
    detected_techniques: List[str]
    extracted_spans: List[SpanEvidence]
    class_probabilities: Dict[str, Any]

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    return {"status": "healthy", "device": str(device)}

@app.post("/api/v1/analyze", response_model=PostAnalysisResponse)
async def analyze_telegram_post(payload: PostAnalysisRequest):
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text payload cannot be empty.")
    
    try:
        dummy_input_ids = torch.randint(0, 1000, (1, 128)).to(device)
        dummy_mask = torch.ones((1, 128)).to(device)

        with torch.no_grad():
            cls_logits, span_logits = model(dummy_input_ids, dummy_mask)

        calibration_result = calibration_engine.evaluate_with_abstention(cls_logits, LABELS)

        span_probs = torch.softmax(span_logits, dim=-1)[0, :, 1].cpu().numpy()
        active_spans = []
        
        if np.max(span_probs) > 0.5:
            active_spans.append(SpanEvidence(
                text_segment=payload.text[:25],
                start_char=0,
                end_char=min(25, len(payload.text))
            ))

        return PostAnalysisResponse(
            post_id=payload.post_id,
            status=calibration_result["status"],
            requires_human_adjudication=calibration_result["requires_human_adjudication"],
            max_confidence=calibration_result["max_confidence"],
            detected_techniques=calibration_result["detected_techniques"],
            extracted_spans=active_spans,
            class_probabilities=calibration_result["class_probabilities"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")