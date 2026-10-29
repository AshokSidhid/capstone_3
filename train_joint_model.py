import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from dataset_loader import UNLPDataset
from models.joint_model import JointManipulationSpanModel
from models.loss_functions import BinaryFocalLoss

def train_and_save():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    output_dir = "weights"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "joint_xlm_roberta_best.pt")

    # 1. Load Model
    model = JointManipulationSpanModel(num_labels=10).to(device)

    for param in model.parameters():
        param.requires_grad = True

    # 2. Loss & Optimizer (2e-5 learning rate prevents mode collapse)
    focal_loss_fn = BinaryFocalLoss(gamma=2.0, alpha=0.25)
    optimizer = AdamW(model.parameters(), lr=2e-5)

    # 3. Data Loader
    train_dataset = UNLPDataset(data_path="data/sample_unlp_data.json", split="train")
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    # 4. Fine-Tuning Loop
    model.train()
    print("\nStarting Balanced Fine-Tuning Pass...")
    
    for epoch in range(5):
        total_loss = 0.0
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            cls_targets = batch["cls_targets"].to(device)

            optimizer.zero_grad()
            
            outputs = model(input_ids, attention_mask)
            cls_logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs

            loss = focal_loss_fn(cls_logits, cls_targets)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/5 | Focal Loss: {avg_loss:.4f}")

    # Save finalized checkpoint
    torch.save(model.state_dict(), save_path)
    print(f"\n--> Successfully saved tuned model weights to: {save_path}")

if __name__ == "__main__":
    train_and_save()