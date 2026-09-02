import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Tuple

class TemperatureScaler(nn.Module):
    """
    Post-hoc Temperature Scaling to calibrate model confidence probabilities.
    Optimizes a single scalar parameter (T) on validation sets to minimize ECE.
    """
    def __init__(self):
        super(TemperatureScaler, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

class ConfidenceCalibrationEngine:
    """
    Calculates Expected Calibration Error (ECE) and provides operational 
    Abstention Guardrails for low-confidence model predictions.
    """
    def __init__(
        self, 
        confidence_threshold: float = 0.65, 
        ece_bins: int = 10
    ):
        self.confidence_threshold = confidence_threshold
        self.ece_bins = ece_bins
        self.scaler = TemperatureScaler()

    def calculate_ece(self, probs: np.ndarray, labels: np.ndarray) -> float:
        """
        Computes Expected Calibration Error (ECE) across probability confidence bins.
        Target requirement: ECE <= 0.08.
        """
        bin_boundaries = np.linspace(0, 1, self.ece_bins + 1)
        ece = 0.0
        total_samples = len(probs)

        for i in range(self.ece_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            # Mask samples belonging to current bin boundary
            in_bin = (probs > bin_lower) & (probs <= bin_upper)
            prop_in_bin = np.mean(in_bin)
            
            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(labels[in_bin])
                avg_confidence_in_bin = np.mean(probs[in_bin])
                ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin

        return float(ece)

    def evaluate_with_abstention(
        self, 
        cls_logits: torch.Tensor, 
        label_names: list
    ) -> Dict[str, Any]:
        """
        Applies temperature calibration and evaluates confidence.
        If maximum confidence < threshold, triggers ABSTAIN status for analyst review.
        """
        # Apply temperature scaling calibration
        calibrated_logits = self.scaler(cls_logits)
        probs = torch.sigmoid(calibrated_logits).detach().cpu().numpy().squeeze(0)
        
        predictions = {}
        flagged_techniques = []
        max_confidence = float(np.max(probs))

        for idx, label in enumerate(label_names):
            prob = float(probs[idx])
            predictions[label] = {
                "probability": round(prob, 4),
                "flagged": prob >= 0.5
            }
            if prob >= 0.5:
                flagged_techniques.append(label)

        # Abstention Decision Logic
        requires_analyst_review = max_confidence < self.confidence_threshold
        status = "ABSTAIN_FLAGGED_FOR_REVIEW" if requires_analyst_review else "VERIFIED_AUTOMATED"

        return {
            "status": status,
            "max_confidence": round(max_confidence, 4),
            "requires_human_adjudication": requires_analyst_review,
            "detected_techniques": flagged_techniques,
            "class_probabilities": predictions
        }