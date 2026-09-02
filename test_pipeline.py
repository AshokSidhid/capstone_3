import torch
from data.dataset_loader import UNLPManipulationDataset
from models.joint_model import JointManipulationSpanModel
from models.loss_functions import MultiTaskUncertaintyLoss

dataset = UNLPManipulationDataset(data_path="data/sample_unlp_data.json")
sample = dataset[0]

print("Input IDs Shape:", sample["input_ids"].shape)
print("Multi-Label Target Vector:", sample["multi_label_targets"])
print("Span Target Labels (Token Bio):", sample["span_targets"][:15])

model = JointManipulationSpanModel(num_labels=10)
input_ids = sample["input_ids"].unsqueeze(0)
attention_mask = sample["attention_mask"].unsqueeze(0)

cls_logits, span_logits = model(input_ids, attention_mask)
print("\nClassification Logits Shape:", cls_logits.shape)
print("Span Extraction Logits Shape:", span_logits.shape)

loss_fn = MultiTaskUncertaintyLoss()
loss = loss_fn(cls_logits, sample["multi_label_targets"].unsqueeze(0), span_logits, sample["span_targets"].unsqueeze(0))
print("Computed Multi-Task Loss:", loss.item())