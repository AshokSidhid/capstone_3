import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Tuple

class TemperatureScaler(nn.Module):
    def __init__(self, temperature: float = 1.0):
        super(TemperatureScaler, self).__init__()
        self.temperature = nn.Parameter(torch.tensor([temperature]))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

class ConfidenceCalibrationEngine:
    def __init__(self, scaler: TemperatureScaler = None, threshold: float = 0.65):
        self.scaler = scaler if scaler is not None else TemperatureScaler()
        self.threshold = threshold

    def calculate_ece(self, probs: np.ndarray, labels: np.ndarray) -> float:
        bin_boundaries = np.linspace(0, 1, self.ece_bins + 1)
        ece = 0.0
        total_samples = len(probs)

        for i in range(self.ece_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            in_bin = (probs > bin_lower) & (probs <= bin_upper)
            prop_in_bin = np.mean(in_bin)
            
            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(labels[in_bin])
                avg_confidence_in_bin = np.mean(probs[in_bin])
                ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin

        return float(ece)

    def evaluate_with_abstention(self, raw_logits: torch.Tensor, label_names: list):
        # Apply temperature scaling
        scaled_logits = self.scaler(raw_logits)
        probs = torch.sigmoid(scaled_logits).squeeze(0).tolist()

        max_conf = max(probs) if probs else 0.0
        requires_review = max_conf < self.threshold

        detected = [
            label for label, prob in zip(label_names, probs) if prob >= 0.5
        ]

        class_probs = {
            label: {"probability": round(prob, 4), "flagged": prob >= 0.5}
            for label, prob in zip(label_names, probs)
        }

        status_str = "ABSTAIN_FLAGGED_FOR_REVIEW" if requires_review else "VERIFIED_AUTOMATED"

        return {
            "status": status_str,
            "max_confidence": round(max_conf, 4),
            "requires_human_adjudication": requires_review,
            "detected_techniques": detected,
            "class_probabilities": class_probs
        }
