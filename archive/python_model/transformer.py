"""
StudentIO Python Multi-Modal Transformer
Handles question answering across text, PDFs, and images
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, ViTImageProcessor, ViTModel
from PIL import Image
import torch
import torch.nn as nn
from typing import List, Dict, Any, Union, Optional
import numpy as np

class MultiModalTransformer:
    """
    Multi-modal transformer for academic Q&A
    
    Processes:
    - Text queries
    - PDF embeddings (pre-extracted text)
    - Image embeddings (via ViT)
    
    Architecture:
   - Text: BART/FLAN-T5 encoder
    - Images: ViT encoder  
    - Cross-attention fusion layer
    - Seq2seq decoder for responses
    """
    
    def __init__(self, 
                 text_model_name: str = "facebook/bart-large",
                 image_model_name: str = "google/vit-base-patch16-224",
                 device: str = "cpu"):
        
        self.device = torch.device(device)
        print(f"🤖 Initializing Multi-Modal Transformer on {device}")
        
        # Text model
        print(f"📝 Loading text model: {text_model_name}")
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.text_model = AutoModelForSeq2SeqLM.from_pretrained(text_model_name).to(self.device)
        
        # Image model  
        print(f"🖼️  Loading image model: {image_model_name}")
        self.image_processor = ViTImageProcessor.from_pretrained(image_model_name)
        self.image_model = ViTModel.from_pretrained(image_model_name).to(self.device)
        
        # Cross-modal fusion layer
        text_dim = self.text_model.config.d_model
        image_dim = self.image_model.config.hidden_size
        
        self.image_projection = nn.Linear(image_dim, text_dim).to(self.device)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=text_dim,
            num_heads=8,
            batch_first=True
        ).to(self.device)
        
        print("✅ Multi-Modal Transformer initialized")
    
    def encode_text(self, text: str) -> torch.Tensor:
        """Encode text to embeddings"""
        inputs = self.text_tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.text_model.model.encoder(**inputs)
            embeddings = outputs.last_hidden_state  # [1, seq_len, dim]
        
        return embeddings
    
    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """Encode image to embeddings"""
        inputs = self.image_processor(image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.image_model(**inputs)
            embeddings = outputs.last_hidden_state  # [1, patches, dim]
        
        # Project to text dimension
        embeddings = self.image_projection(embeddings)
        
        return embeddings
    
    def fuse_modalities(self,
                        text_emb: torch.Tensor,
                        image_embs: Optional[List[torch.Tensor]] = None,
                        pdf_embs: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        """
        Fuse multi-modal embeddings via cross-attention
        
        Args:
            text_emb: Text query embeddings [1, seq_len, dim]
            image_embs: List of image embeddings
            pdf_embs: List of PDF text embeddings
            
        Returns:
            Fused embeddings [1, seq_len, dim]
        """
        # Start with text as query
        fused = text_emb
        
        # Cross-attend to images
        if image_embs:
            for img_emb in image_embs:
                fused, _ = self.cross_attention(
                    query=fused,
                    key=img_emb,
                    value=img_emb
                )
        
        # Cross-attend to PDFs
        if pdf_embs:
            for pdf_emb in pdf_embs:
                fused, _ = self.cross_attention(
                    query=fused,
                    key=pdf_emb,
                    value=pdf_emb
                )
        
        return fused
    
    def generate_response(self,
                         query: str,
                         images: Optional[List[Image.Image]] = None,
                         pdf_texts: Optional[List[str]] = None,
                         max_length: int = 512) -> Dict[str, Any]:
        """
        Generate a response to a multi-modal query
        
        Args:
            query: Text question
            images: List of PIL Images
            pdf_texts: List of extracted PDF texts
            max_length: Max response length
            
        Returns:
            Dict with answer, confidence, sources
        """
        # Encode text query
        text_emb = self.encode_text(query)
        
        # Encode images
        image_embs = []
        if images:
            for img in images:
                image_embs.append(self.encode_image(img))
        
        # Encode PDFs
        pdf_embs = []
        if pdf_texts:
            for pdf_text in pdf_texts:
                pdf_embs.append(self.encode_text(pdf_text))
        
        # Fuse modalities
        fused_emb = self.fuse_modalities(text_emb, image_embs, pdf_embs)
        
        # Generate response
        # Create decoder inputs
        decoder_input_ids = self.text_tokenizer(
            "",
            return_tensors="pt"
        ).input_ids.to(self.device)
        
        # Generate with fused embeddings as encoder output
        with torch.no_grad():
            outputs = self.text_model.generate(
                encoder_outputs=(fused_emb,),
                decoder_input_ids=decoder_input_ids,
                max_length=max_length,
                num_beams=4,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.2,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        # Decode response
        answer = self.text_tokenizer.decode(
            outputs.sequences[0],
            skip_special_tokens=True
        )
        
        # Calculate confidence
        if hasattr(outputs, 'sequences_scores'):
            confidence = torch.exp(outputs.sequences_scores[0]).item()
        else:
            confidence = 0.85
        
        return {
            "answer": answer,
            "confidence": confidence,
            "modalities_used": {
                "text": True,
                "images": len(image_embs) if images else 0,
                "pdfs": len(pdf_embs) if pdf_texts else 0
            },
            "sources": []
        }
    
    def get_embeddings(self,
                      text: Optional[str] = None,
                      image: Optional[Image.Image] = None) -> np.ndarray:
        """
        Get embeddings for meta-learning integration
        
        Returns mean-pooled vector for Julia backend
        """
        embeddings = []
        
        if text:
            text_emb = self.encode_text(text)
            embeddings.append(text_emb.mean(dim=1))  # Mean pool over sequence
        
        if image:
            img_emb = self.encode_image(image)
            embeddings.append(img_emb.mean(dim=1))  # Mean pool over patches
        
        if not embeddings:
            raise ValueError("Must provide at least one modality")
        
        # Concatenate and return as numpy
        final_emb = torch.cat(embeddings, dim=1)
        return final_emb.cpu().numpy()


def create_default_model(device: str = "cpu") -> MultiModalTransformer:
    """Factory function to create model with defaults"""
    return MultiModalTransformer(device=device)
