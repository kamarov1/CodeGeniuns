import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, RobertaTokenizer, AdamW, get_linear_schedule_with_warmup
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
import logging
import os
import json
from tqdm import tqdm
import numpy as np
from torch.cuda.amp import autocast, GradScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class CodeT5Config:
    model_name: str = "Salesforce/codet5-base"
    max_length: int = 512
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    num_epochs: int = 5
    gradient_accumulation_steps: int = 2
    max_grad_norm: float = 1.0
    use_fp16: bool = True
    seed: int = 42
    num_workers: int = 4
    eval_batch_size: int = 32
    save_steps: int = 1000
    eval_steps: int = 500

class CodeDataset(Dataset):
    def __init__(self, data: List[Dict[str, str]], tokenizer: RobertaTokenizer, max_length: int):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        inputs = self.tokenizer.encode_plus(
            item["prompt"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        targets = self.tokenizer.encode_plus(
            item["code"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": inputs.input_ids.squeeze(),
            "attention_mask": inputs.attention_mask.squeeze(),
            "labels": targets.input_ids.squeeze(),
        }

class CodeT5Model:
    def __init__(self, config: CodeT5Config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        
        self.model = T5ForConditionalGeneration.from_pretrained(config.model_name).to(self.device)
        self.tokenizer = RobertaTokenizer.from_pretrained(config.model_name)
        self.scaler = GradScaler(enabled=config.use_fp16)

    def train(self, train_data: List[Dict[str, str]], val_data: Optional[List[Dict[str, str]]] = None):
        train_dataset = CodeDataset(train_data, self.tokenizer, self.config.max_length)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True, 
            num_workers=self.config.num_workers,
            pin_memory=True
        )

        optimizer = AdamW(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        total_steps = len(train_loader) * self.config.num_epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

        global_step = 0
        for epoch in range(self.config.num_epochs):
            self.model.train()
            epoch_loss = 0
            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                with autocast(enabled=self.config.use_fp16):
                    outputs = self.model(**batch)
                    loss = outputs.loss / self.config.gradient_accumulation_steps

                self.scaler.scale(loss).backward()
                epoch_loss += loss.item()

                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                if global_step % self.config.save_steps == 0:
                    self.save_model(f"checkpoint-{global_step}")

                if val_data and global_step % self.config.eval_steps == 0:
                    val_loss = self.evaluate(val_data)
                    logger.info(f"Step {global_step}, Validation Loss: {val_loss:.4f}")
                    self.model.train()

            avg_train_loss = epoch_loss / len(train_loader)
            logger.info(f"Epoch {epoch+1}/{self.config.num_epochs}, Average Train Loss: {avg_train_loss:.4f}")

        logger.info("Training completed")

    def evaluate(self, data: List[Dict[str, str]]) -> float:
        eval_dataset = CodeDataset(data, self.tokenizer, self.config.max_length)
        eval_loader = DataLoader(
            eval_dataset, 
            batch_size=self.config.eval_batch_size, 
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                total_loss += outputs.loss.item()
        
        return total_loss / len(eval_loader)

    @torch.no_grad()
    def generate_code(self, prompt: str, max_length: Optional[int] = None, num_return_sequences: int = 1) -> List[str]:
        max_length = max_length or self.config.max_length
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        outputs = self.model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            no_repeat_ngram_size=2,
            num_beams=4,
            early_stopping=True,
            temperature=0.7,
            top_p=0.95,
            do_sample=True
        )

        return [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

    def save_model(self, path: str):
        if not os.path.exists(path):
            os.makedirs(path)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(self.config.__dict__, f)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        with open(os.path.join(path, "config.json"), "r") as f:
            config_dict = json.load(f)
        self.config = CodeT5Config(**config_dict)
        self.model = T5ForConditionalGeneration.from_pretrained(path).to(self.device)
        self.tokenizer = RobertaTokenizer.from_pretrained(path)
        logger.info(f"Model loaded from {path}")

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model.config.model_type,
            "model_size": sum(p.numel() for p in self.model.parameters()),
            "vocab_size": self.tokenizer.vocab_size,
            "device": str(self.device),
            "config": self.config.__dict__
        }

    def predict(self, prompt: str, max_length: Optional[int] = None) -> Dict[str, Union[str, float]]:
        generated_code = self.generate_code(prompt, max_length)[0]
        
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids, decoder_input_ids=input_ids)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            confidence = torch.mean(torch.max(probs, dim=-1)[0]).item()

        return {
            "generated_code": generated_code,
            "confidence": confidence
        }

    def batch_predict(self, prompts: List[str], max_length: Optional[int] = None) -> List[Dict[str, Union[str, float]]]:
        return [self.predict(prompt, max_length) for prompt in prompts]
