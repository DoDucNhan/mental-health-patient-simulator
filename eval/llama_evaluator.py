# llama_evaluator.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

class LlamaEvaluator:
    def __init__(self, model_name="meta-llama/Llama-3.1-70B"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
    def evaluate_response(self, question, response):
        prompt = self.create_evaluation_prompt(question, response)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=2048,
                temperature=0.1,  # Low temperature for consistent evaluation
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        evaluation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        evaluation = evaluation[len(prompt):].strip()
        
        return self.parse_scores(evaluation)
    
    def create_evaluation_prompt(self, question, response):
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert mental health professional evaluating AI-simulated patient responses. Rate each response on the following metrics (1-10 scale):

1. **Active Listening** (1-10): How well the patient demonstrates understanding
2. **Emotional Expression** (1-10): How authentically emotions are expressed  
3. **Clinical Realism** (1-10): How realistic the patient presentation is
4. **Conversational Flow** (1-10): How natural the conversation feels
5. **Symptom Accuracy** (1-10): How accurately symptoms are portrayed
6. **Educational Value** (1-10): How useful for training purposes
7. **Overall Quality** (1-10): Overall assessment

Provide scores in this exact format:
Active Listening: [score]
Emotional Expression: [score]  
Clinical Realism: [score]
Conversational Flow: [score]
Symptom Accuracy: [score]
Educational Value: [score]
Overall Quality: [score]<|eot_id|><|start_header_id|>user<|end_header_id|>

Therapist Question: {question}

Patient Response: {response}

Please evaluate this response:<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

    def parse_scores(self, evaluation_text):
        metrics = [
            "Active Listening", "Emotional Expression", "Clinical Realism",
            "Conversational Flow", "Symptom Accuracy", "Educational Value", "Overall Quality"
        ]
        
        scores = {}
        for metric in metrics:
            import re
            pattern = f"{metric}: (\d+)"
            match = re.search(pattern, evaluation_text, re.IGNORECASE)
            if match:
                scores[metric.lower().replace(" ", "_")] = int(match.group(1))
            else:
                scores[metric.lower().replace(" ", "_")] = 5  # Default score
        
        return scores