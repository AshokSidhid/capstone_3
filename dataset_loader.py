import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class D(Dataset):
    def __init__(self, data_path="data/sample_unlp_data.json", split="train", max_length=512):
        self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
        self.max_length = max_length

        # Explicitly balanced dataset: Positive (Propaganda) vs Negative (Neutral/Weather/News)
        self.data = [
            # --- POSITIVE EXAMPLES (Propaganda / Manipulation) ---
            {
                "text": "Це ганебна брехня та підла маніпуляція ворожих агентів, які намагаються залякати народи Євразії!",
                "cls_targets": [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            },
            {
                "text": "Наші відважні воїни знищують підлого ворога! Перемога за нами, ворог буде знищений!",
                "cls_targets": [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
            },
            {
                "text": "Злочинний режим готує таємний хімічний удар проти мирного населення вже сьогодні ввечері!",
                "cls_targets": [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            },
            
            # --- NEGATIVE EXAMPLES (Neutral / Weather / Fact-Based) ---
            {
                "text": "Завтра в північних областях очікується хмарна погода з короткочасними дощами та температурою до +18°C.",
                "cls_targets": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            },
            {
                "text": "Міністерство енергетики повідомляє про планові ремонтні роботи на електромережах у четвер.",
                "cls_targets": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            },
            {
                "text": "За даними Державної служби статистики, обсяг експорту сільськогосподарської продукції зріс на 2,4%.",
                "cls_targets": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            }
        ] * 10  # Multiply to create a sufficient training batch pool

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]

        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "cls_targets": torch.tensor(item["cls_targets"], dtype=torch.float),
            "span_targets": torch.zeros(self.max_length, dtype=torch.long)
        }