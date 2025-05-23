# mistral_evaluator.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class MistralEvaluator:
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.3"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
    def evaluate_response(self, question, response):
        prompt = f"""[INST] You are evaluating an AI simulating a mental health patient. 

Rate this patient response on a scale of 1-10 for each metric:
- Realism: How realistic does this sound as a real patient?
- Emotional_Expression: How well are emotions conveyed?
- Clinical_Accuracy: How clinically accurate is the presentation?
- Conversational_Quality: How natural is the conversation?
- Training_Value: How valuable for training mental health professionals?

Therapist: {question}
Patient: {response}

Provide scores in format: Realism: X, Emotional_Expression: X, Clinical_Accuracy: X, Conversational_Quality: X, Training_Value: X [/INST]"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=1024,
                temperature=0.1,
                do_sample=True
            )
        
        evaluation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        evaluation = evaluation.split("[/INST]")[-1].strip()
        
        return self.parse_scores(evaluation)
    
    def parse_scores(self, evaluation_text):
        import re
        scores = {}
        metrics = ["Realism", "Emotional_Expression", "Clinical_Accuracy", 
                  "Conversational_Quality", "Training_Value"]
        
        for metric in metrics:
            pattern = f"{metric}: (\d+)"
            match = re.search(pattern, evaluation_text)
            if match:
                scores[metric.lower()] = int(match.group(1))
            else:
                scores[metric.lower()] = 5
        
        return scores