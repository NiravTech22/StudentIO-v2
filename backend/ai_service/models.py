"""
AI Model Management
Handles loading and running Hugging Face transformer models
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import torch
from typing import Tuple, AsyncGenerator, Optional
import asyncio


class AIModel:
    """Wrapper for Hugging Face models with streaming support"""
    
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.tokenizer = None
        self.model = None
        
    def load(self):
        """Load the model and tokenizer"""
        print(f"Loading tokenizer for {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        print(f"Loading model {self.model_name}...")
        
        # Try to load as seq2seq first (FLAN-T5, T5, BART)
        try:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32 if self.device == "cpu" else torch.float16
            )
            self.is_seq2seq = True
        except:
            # Fall back to causal LM (GPT-2, GPT-Neo, Llama)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32 if self.device == "cpu" else torch.float16
            )
            self.is_seq2seq = False
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ Model loaded on {self.device}")
    
    def generate(self, prompt: str, max_length: int = 512) -> Tuple[str, float]:
        """
        Generate a complete answer for the given prompt
        Returns (answer, confidence)
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded")
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self. device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                early_stopping=True,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        # Decode output
        generated_text = self.tokenizer.decode(
            outputs.sequences[0],
            skip_special_tokens=True
        )
        
        # Calculate confidence (average of token scores)
        if hasattr(outputs, 'sequences_scores'):
            confidence = torch.exp(outputs.sequences_scores[0]).item()
        else:
            confidence = 0.85  # Default confidence
        
        return generated_text, confidence
    
    async def generate_stream(
        self,
        prompt: str,
        max_length: int = 512
    ) -> AsyncGenerator[str, None]:
        """
        Stream generate tokens one at a time
        Yields individual tokens as they're generated
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded")
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.device)
        
        # Generate tokens iteratively
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        
        for _ in range(max_length):
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                if self.is_seq2seq:
                    # For seq2seq models, use encoder-decoder
                    decoder_input_ids = torch.tensor([[self.tokenizer.pad_token_id]]).to(self.device)
                    outputs = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=1,
                        num_beams=1,
                        do_sample=True,
                        temperature=0.7
                    )
                    next_token_id = outputs[0, -1].item()
                else:
                    # For causal LM, get next token from logits
                    logits = outputs.logits[0, -1, :]
                    probs = torch.softmax(logits, dim=-1)
                    next_token_id = torch.multinomial(probs, num_samples=1).item()
            
            # Check for end of sequence
            if next_token_id == self.tokenizer.eos_token_id:
                break
            
            # Decode and yield token
            token_text = self.tokenizer.decode([next_token_id], skip_special_tokens=True)
            
            if token_text:  # Only yield non-empty tokens
                yield token_text
                await asyncio.sleep(0)  # Allow other tasks to run
            
            # Update input for next iteration
            next_token_tensor = torch.tensor([[next_token_id]]).to(self.device)
            input_ids = torch.cat([input_ids, next_token_tensor], dim=1)
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((1, 1), dtype=torch.long, device=self.device)
            ], dim=1)


def load_model(model_name: str, device: str = "cpu") -> AIModel:
    """
    Factory function to load and return an AI model
    
    Recommended models:
    - "google/flan-t5-small" - Fast, good for CPU (80M params)
    - "google/flan-t5-base" - Balanced (250M params)
    - "google/flan-t5-large" - Better quality (780M params, needs GPU)
    - "gpt2" - Alternative causal LM (124M params)
    - "gpt2-medium" - Larger GPT-2 (355M params)
    """
    model = AIModel(model_name, device)
    model.load()
    return model


def get_recommended_model(device: str = "cpu") -> str:
    """Get recommended model based on available hardware"""
    if device == "cuda":
        return "google/flan-t5-large"  # Better model for GPU
    else:
        return "google/flan-t5-base"  # Good balance for CPU
