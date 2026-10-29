import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        probs = torch.sigmoid(inputs)
        
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        focal_weight = (1.0 - p_t) ** self.gamma
        
        alpha_weight = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        loss = alpha_weight * focal_weight * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

class MultiTaskUncertaintyLoss(nn.Module):
    def __init__(self, num_tasks=2):
        super().__init__()
        # Learnable log variances log(sigma^2) for each task
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, loss_cls, loss_span):
        # Homoscedastic uncertainty loss formulation:
        # L_total = exp(-log_var_1) * loss_1 + log_var_1 + exp(-log_var_2) * loss_2 + log_var_2
        precision_cls = torch.exp(-self.log_vars[0])
        precision_span = torch.exp(-self.log_vars[1])
        
        total_loss = (precision_cls * loss_cls + self.log_vars[0]) + \
                     (precision_span * loss_span + self.log_vars[1])
        return total_loss
