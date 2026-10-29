import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import List, Dict, Any, Tuple

class UNLPManipulationDataset(Dataset):
    def __init__(
        self, 
        data_path: str, 
        tokenizer_name: str = "xlm-roberta-base", 
        max_length: int = 512,
        label_list: List[str] = None
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.label_list = label_list or [
            "Appeal_to_Authority", "Bandwagon", "Causal_Oversimplification", 
            "Doubt", "Emotional_Charge", "Loaded_Language", "Name_Calling", 
            "Repetition", "Slogans", "Straw_Man"
        ]
        self.label2id = {label: i for i, label in enumerate(self.label_list)}
        
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def __len__(self) -> int:
        return len(self.data)

    def _align_spans_to_tokens(
        self, 
        text: str, 
        spans: List[Dict[str, int]], 
        offset_mapping: List[Tuple[int, int]]
    ) -> List[int]:
        bio_labels = [0] * len(offset_mapping)
        
        for span in spans:
            start_char = span["start"]
            end_char = span["end"]
            
            for token_idx, (token_start, token_end) in enumerate(offset_mapping):
                if token_start == 0 and token_end == 0:
                    continue
                
                if token_start >= start_char and token_end <= end_char:
                    bio_labels[token_idx] = 1

        return bio_labels

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        text = item["text"]
        
        multi_label_vec = [0.0] * len(self.label_list)
        for technique in item.get("techniques", []):
            if technique in self.label2id:
                multi_label_vec[self.label2id[technique]] = 1.0

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_offsets_mapping=True,
            return_tensors="pt"
        )

        offset_mapping = encoding["offset_mapping"].squeeze(0).tolist()
        span_labels = self._align_spans_to_tokens(
            text, 
            item.get("evidence_spans", []), 
            offset_mapping
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "multi_label_targets": torch.tensor(multi_label_vec, dtype=torch.float),
            "span_targets": torch.tensor(span_labels, dtype=torch.long)
        }
