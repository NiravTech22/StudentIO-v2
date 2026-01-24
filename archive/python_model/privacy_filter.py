"""
Privacy Filter - Academic Content Classifier
Filters non-academic content before meta-learning
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import Dict, Tuple
import re

class PrivacyFilter:
    """
    Lightweight classifier to detect academic vs non-academic content
    
    Academic categories:
    - Educational questions
    - Academic discussion
    - Study material
    - Research queries
    
    Non-academic (filtered):
    - Personal conversations
    - Non-educational content
    - Inappropriate material
    """
    
    def __init__(self, threshold: float = 0.7, device: str = "cpu"):
        self.threshold = threshold
        self.device = torch.device(device)
        
        # Use a lightweight classifier (DistilBERT)
        model_name = "distilbert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # In production, use fine-tuned model
        # For now, use heuristics + simple model
        self.academic_keywords = {
            'explain', 'how', 'what', 'why', 'when', 'where',
            'calculate', 'solve', 'prove', 'demonstrate',
            'study', 'learn', 'understand', 'concept',
            'theory', 'principle', 'formula', 'equation',
            'algorithm', 'method', 'technique', 'process'
        }
        
        self.non_academic_keywords = {
            'personal', 'private', 'secret', 'password',
            'hack', 'cheat', 'plagiarize', 'copy'
        }
        
        print("✅ Privacy Filter initialized")
    
    def is_academic(self, text: str) -> Tuple[bool, float, str]:
        """
        Classify if content is academic
        
        Returns:
            (is_academic, confidence, reason)
        """
        text_lower = text.lower()
        
        # Quick heuristic checks
        academic_score = sum(1 for kw in self.academic_keywords if kw in text_lower)
        non_academic_score = sum(1 for kw in self.non_academic_keywords if kw in text_lower)
        
        # Question patterns (likely academic)
        question_patterns = [
            r'\bwhat\s+is\b',
            r'\bhow\s+(do|does|can)\b',
            r'\bwhy\s+',
            r'\bexplain\b',
            r'\bcalculate\b',
            r'\bsolve\b'
        ]
        
        has_question_pattern = any(re.search(pat, text_lower) for pat in question_patterns)
        
        # Scoring
        confidence = 0.5  # Base
        
        if academic_score > 2 or has_question_pattern:
            confidence += 0.3
        
        if non_academic_score > 0:
            confidence -= 0.4
        
        # Check length (very short might not be academic)
        if len(text.split()) < 3:
            confidence -= 0.2
        
        # Mathematical content detection
        if re.search(r'[0-9]+\s*[+\-*/=]\s*[0-9]+', text):
            confidence += 0.2
        
        # Final decision
        is_academic = confidence >= self.threshold
        
       reason = ""
        if is_academic:
            if has_question_pattern:
                reason = "question_pattern_detected"
            elif academic_score > 0:
                reason = f"academic_keywords_{academic_score}"
            else:
                reason = "general_academic_context"
        else:
            if non_academic_score > 0:
                reason = "non_academic_keywords_detected"
            elif len(text.split()) < 3:
                reason = "content_too_short"
            else:
                reason = "insufficient_academic_indicators"
        
        return is_academic, min(max(confidence, 0.0), 1.0), reason
    
    def filter_batch(self, texts: list) -> Dict[str, list]:
        """
        Filter a batch of texts
        
        Returns:
            {
                'academic': [...],
                'filtered': [...],
                'confidences': [...]
            }
        """
        academic = []
        filtered = []
        confidences = []
        
        for text in texts:
            is_acad, conf, reason = self.is_academic(text)
            confidences.append(conf)
            
            if is_acad:
                academic.append(text)
            else:
                filtered.append({"text": text, "reason": reason})
        
        return {
            "academic": academic,
            "filtered": filtered,
            "confidences": confidences
        }
    
    def sanitize_text(self, text: str) -> str:
        """
        Remove potentially sensitive information
        
        - Email addresses
        - Phone numbers
        - URLs (keep educational domains)
        """
        # Remove emails
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
        
        # Remove phone numbers
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)
        
        # Remove non-educational URLs
        educational_domains = ['edu', 'ac.uk', 'wikipedia', 'arxiv', 'scholar']
        
        def url_filter(match):
            url = match.group(0)
            if any(domain in url for domain in educational_domains):
                return url
            return '[URL]'
        
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                     url_filter, text)
        
        return text


def create_filter(threshold: float = 0.7) -> PrivacyFilter:
    """Factory function"""
    return PrivacyFilter(threshold=threshold)
