import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Dict, Any, Tuple
from collections import OrderedDict
from tqdm import tqdm
import logging
from transformers import T5ForConditionalGeneration, RobertaTokenizer
from torch.optim import Adam
from torch.cuda.amp import autocast, GradScaler

logger = logging.getLogger(__name__)

class MAMLCodeT5:
    def __init__(self, model: T5ForConditionalGeneration, tokenizer: RobertaTokenizer, 
                 device: torch.device, config: Dict[str, Any]):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.config = config
        self.meta_optimizer = Adam(self.model.parameters(), lr=config['meta_lr'])
        self.scaler = GradScaler(enabled=config['use_fp16'])

    def inner_loop(self, support_set: List[Dict[str, str]]) -> OrderedDict:
        self.model.train()
        fast_weights = OrderedDict(self.model.named_parameters())

        dataloader = DataLoader(support_set, batch_size=self.config['inner_batch_size'], shuffle=True)
        
        for _ in range(self.config['num_inner_steps']):
            for batch in dataloader:
                inputs = self.tokenizer(batch['prompt'], return_tensors='pt', padding=True, truncation=True, max_length=self.config['max_length']).to(self.device)
                labels = self.tokenizer(batch['code'], return_tensors='pt', padding=True, truncation=True, max_length=self.config['max_length']).to(self.device).input_ids

                with autocast(enabled=self.config['use_fp16']):
                    outputs = self.model(**inputs, labels=labels)
                    loss = outputs.loss

                grads = torch.autograd.grad(loss, fast_weights.values(), create_graph=not self.config['first_order'])
                fast_weights = OrderedDict(
                    (name, param - self.config['inner_lr'] * grad)
                    for ((name, param), grad) in zip(fast_weights.items(), grads)
                )

        return fast_weights

    def outer_loop(self, tasks: List[Dict[str, List[Dict[str, str]]]]) -> float:
        meta_loss = 0.0
        for task in tasks:
            support_set = task['support']
            query_set = task['query']

            fast_weights = self.inner_loop(support_set)

            # Evaluate on query set
            self.model.eval()
            with torch.no_grad():
                query_dataloader = DataLoader(query_set, batch_size=self.config['outer_batch_size'])
                for batch in query_dataloader:
                    inputs = self.tokenizer(batch['prompt'], return_tensors='pt', padding=True, truncation=True, max_length=self.config['max_length']).to(self.device)
                    labels = self.tokenizer(batch['code'], return_tensors='pt', padding=True, truncation=True, max_length=self.config['max_length']).to(self.device).input_ids

                    with autocast(enabled=self.config['use_fp16']):
                        outputs = self.model(**inputs, labels=labels)
                        meta_loss += outputs.loss

        meta_loss /= len(tasks)
        return meta_loss

    def train(self, tasks: List[Dict[str, List[Dict[str, str]]]], num_iterations: int) -> None:
        for iteration in tqdm(range(num_iterations), desc="Meta-training"):
            self.meta_optimizer.zero_grad()
            meta_loss = self.outer_loop(tasks)
            
            self.scaler.scale(meta_loss).backward()
            self.scaler.unscale_(self.meta_optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config['max_grad_norm'])
            self.scaler.step(self.meta_optimizer)
            self.scaler.update()

            if iteration % self.config['log_interval'] == 0:
                logger.info(f"Iteration {iteration}, Meta Loss: {meta_loss.item():.4f}")

    def adapt(self, support_set: List[Dict[str, str]]) -> None:
        fast_weights = self.inner_loop(support_set)
        self.model.load_state_dict(fast_weights)

    @torch.no_grad()
    def generate(self, prompt: str) -> str:
        self.model.eval()
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        output_ids = self.model.generate(
            input_ids, 
            max_length=self.config['max_length'],
            num_beams=self.config['num_beams'],
            no_repeat_ngram_size=self.config['no_repeat_ngram_size'],
            top_k=self.config['top_k'],
            top_p=self.config['top_p'],
            temperature=self.config['temperature']
        )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

def prepare_tasks(data: List[Dict[str, str]], num_tasks: int, support_size: int, query_size: int) -> List[Dict[str, List[Dict[str, str]]]]:
    import random
    random.shuffle(data)
    
    tasks = []
    for _ in range(num_tasks):
        task_data = random.sample(data, support_size + query_size)
        tasks.append({
            'support': task_data[:support_size],
            'query': task_data[support_size:]
        })
    
    return tasks

def load_data(file_path: str) -> List[Dict[str, str]]:
    import json
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def save_model(model: MAMLCodeT5, path: str) -> None:
    torch.save({
        'model_state_dict': model.model.state_dict(),
        'config': model.config
    }, path)
    logger.info(f"Model saved to {path}")

def load_model(model: T5ForConditionalGeneration, tokenizer: RobertaTokenizer, path: str, device: torch.device) -> MAMLCodeT5:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    config = checkpoint['config']
    maml = MAMLCodeT5(model, tokenizer, device, config)
    logger.info(f"Model loaded from {path}")
    return maml
