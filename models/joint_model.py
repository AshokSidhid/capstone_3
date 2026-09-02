import torch
import torch.nn as nn
from transformers import AutoModel, PreTrainedModel

class JointManipulationSpanModel(nn.Module):
    def __init__(
        self, 
        model_name: str = "xlm-roberta-base", 
        num_labels: int = 10, 
        dropout_prob: float = 0.2
    ):
        super(JointManipulationSpanModel, self).__init__()
        
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        
        self.dropout = nn.Dropout(dropout_prob)
        
        self.classifier_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size // 2, num_labels)
        )
        
        self.span_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size // 2, 2)
        )

    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        sequence_output = outputs.last_hidden_state
        pooled_output = sequence_output[:, 0, :]

        pooled_output = self.dropout(pooled_output)
        sequence_output = self.dropout(sequence_output)

        cls_logits = self.classifier_head(pooled_output)
        span_logits = self.span_head(sequence_output)

        return cls_logits, span_logits