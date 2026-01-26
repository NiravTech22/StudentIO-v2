
import sys
import os

print(f"Python executable: {sys.executable}")
print(f"Current working directory: {os.getcwd()}")

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    print("✅ Transformers/Torch imported successfully")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

MODEL_NAME = "google/flan-t5-large"

def test_generation():
    print(f"Loading {MODEL_NAME}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        print("✅ Model loaded")
    except Exception as e:
        print(f"❌ Model load failed: {e}")
        return

    questions = [
        "what is 2 + 2?",
        "Calculate the eigenvalue of a specific 2x2 matrix",
        "How do I solve an integral?",
        "Explain the 2008 financial crisis"
    ]

    for q in questions:
        print(f"\n==================================================")
        print(f"QUESTION: {q}")
        
        # Strategy A: Current Complex System Prompt
        prompt_complex = (
            "Instruction: You are an expert professor. Answer with deep reasoning. "
            "Think step-by-step. Do not use circular logic. "
            f"Question: {q}\nAnswer:"
        )
        
        # Strategy B: Simple Zero-Shot CoT
        prompt_simple = f"Question: {q}\nLet's think step by step.\nAnswer:"
        
        # Strategy C: Direct
        prompt_direct = f"Q: {q}\nA:"

        strategies = [
            ("Complex", prompt_complex),
            ("Simple CoT", prompt_simple),
            ("Direct", prompt_direct)
        ]

        for name, prompt in strategies:
            inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            
            outputs = model.generate(
                **inputs,
                max_length=200,
                num_beams=5,
                temperature=0.3, # Slightly relaxed
                do_sample=True,
                repetition_penalty=1.2,
                early_stopping=True
            )
            
            ans = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"👉 [{name}]: {ans}")

if __name__ == "__main__":
    test_generation()
