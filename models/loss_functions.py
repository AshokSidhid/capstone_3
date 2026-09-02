import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryFocalLoss(nn.Module):
    """
    Focal Loss for Multi-Label Binary Classification.
    Down-weights easy positive/negative examples and forces gradients to focus 
    on hard, rare manipulation classes.
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Compute standard binary cross entropy loss per element
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        probs = torch.sigmoid(inputs)
        
        # Calculate pt (probability of target class)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # Calculate focal weight factor
        focal_weight = (1.0 - p_t) ** self.gamma
        
        # Apply alpha weighting
        alpha_weight = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        loss = alpha_weight * focal_weight * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

class MultiTaskUncertaintyLoss(nn.Module):
    """
    Learns dynamic homoscedastic uncertainty weights (sigma_1, sigma_2) to balance:
    1. Multi-Label Classification Loss (Focal Loss)
    2. Span Extraction Loss (Cross Entropy)
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super(MultiTaskUncertaintyLoss, self).__init__()
        self.focal_loss = BinaryFocalLoss(alpha=alpha, gamma=gamma)
        self.span_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        
        # Learnable log variances for dynamic loss balancing
        self.log_var_cls = nn.Parameter(torch.zeros(1))
        self.log_var_span = nn.Parameter(torch.zeros(1))

    def forward(
        self, 
        cls_logits: torch.Tensor, 
        cls_targets: torch.Tensor, 
        span_logits: torch.Tensor, 
        span_targets: torch.Tensor
    ) -> torch.Tensor:
        # Calculate individual task losses
        l_cls = self.focal_loss(cls_logits, cls_targets)
        
        # Reshape span tensors for CrossEntropy Loss [batch_size * seq_len, num_classes]
        span_logits_flat = span_logits.view(-1, 2)
        span_targets_flat = span_targets.view(-1)
        l_span = self.span_loss_fn(span_logits_flat, span_targets_flat)
        
        # Homoscedastic uncertainty weighting equation:
        # Loss_total = exp(-log_var1) * L1 + log_var1 + exp(-log_var2) * L2 + log_var2
        precision_cls = torch.exp(-self.log_var_cls)
        precision_span = torch.exp(-self.log_var_span)
        
        total_loss = (
            precision_cls * l_cls + self.log_var_cls +
            precision_span * l_span + self.log_var_span
        )
        return total_loss