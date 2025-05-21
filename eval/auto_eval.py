import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import openai
import google.generativeai as genai

# Set up API keys
openai.api_key = os.environ.get("OPENAI_API_KEY")
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

# Load fine-tuned models
def load_model(model_name, model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    return model, tokenizer

# Dictionary of fine-tuned models
models = {
    "OpenHermes-2.5-Mistral": load_model("OpenHermes-2.5-Mistral", "path/to/finetuned/openhermes"),
    "Phi-3-mini": load_model("Phi-3-mini", "path/to/finetuned/phi3mini"),
    "Gemma-7B-Instruct": load_model("Gemma-7B-Instruct", "path/to/finetuned/gemma7b"),
    "Llama-3-8B-Instruct": load_model("Llama-3-8B-Instruct", "path/to/finetuned/llama3_8b")
}

# Function to generate responses from models
def generate_response(model, tokenizer, prompt, max_length=1024):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_length,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the model's response part (after the prompt)
    response = response[len(prompt):].strip()
    
    return response

# Load test questions
def load_test_questions(file_path):
    with open(file_path, 'r') as f:
        questions = json.load(f)
    return questions

# Evaluation prompt templates
def get_evaluation_prompt(question, response):
    prompt = f"""Please act as an impartial judge and evaluate the quality of the response provided by an AI simulating a mental health patient with depression or anxiety to the therapist question displayed below.

Your evaluation should be based solely on the following mental health simulation metrics:
1. Active Listening: How well the simulated patient demonstrates understanding of the therapist's questions without making assumptions.
2. Empathy & Validation: How authentically the patient expresses emotions and validates the therapeutic relationship.
3. Safety & Trustworthiness: How consistently the patient portrays a clinically accurate presentation without harmful content.
4. Open-mindedness & Non-judgment: How well the patient shows realistic response to therapeutic input without unrealistic bias.
5. Maladaptive Cognitions: How accurately the patient portrays cognitive distortions typical in depression/anxiety.
6. Emotional States: How realistically the patient expresses emotions associated with depression/anxiety.
7. Conversational Style: How well the patient mimics natural human conversation patterns rather than idealized responses.

Be objective and provide a score for EACH metric on a scale of 1-10, where:
1: Completely unrealistic, no resemblance to real patient behavior
5: Moderately realistic but with noticeable artificial elements
10: Indistinguishable from a real patient with depression/anxiety

Therapist Question: {question}

Patient Response: {response}

Evaluation:
"""
    return prompt

# Function to evaluate responses using GPT-4
def evaluate_with_gpt4(question, response):
    prompt = get_evaluation_prompt(question, response)
    try:
        completion = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        evaluation = completion.choices[0].message.content
        
        # Parse the scores from the evaluation text
        scores = parse_evaluation_scores(evaluation)
        return scores
    except Exception as e:
        print(f"Error with GPT-4 evaluation: {e}")
        return None

# Function to evaluate responses using Gemini Pro
def evaluate_with_gemini(question, response):
    prompt = get_evaluation_prompt(question, response)
    try:
        gemini_model = genai.GenerativeModel('gemini-pro')
        response = gemini_model.generate_content(prompt)
        evaluation = response.text
        
        # Parse the scores from the evaluation text
        scores = parse_evaluation_scores(evaluation)
        return scores
    except Exception as e:
        print(f"Error with Gemini evaluation: {e}")
        return None

# Function to parse evaluation scores from text
def parse_evaluation_scores(evaluation_text):
    metrics = [
        "Active Listening", "Empathy & Validation", "Safety & Trustworthiness",
        "Open-mindedness & Non-judgment", "Maladaptive Cognitions", 
        "Emotional States", "Conversational Style"
    ]
    
    scores = {}
    for metric in metrics:
        # Look for patterns like "Active Listening: 8/10" or "Active Listening - 8"
        patterns = [
            f"{metric}: (\d+)",
            f"{metric} - (\d+)",
            f"{metric}: (\d+)/10",
            f"{metric} score: (\d+)"
        ]
        
        for pattern in patterns:
            import re
            match = re.search(pattern, evaluation_text, re.IGNORECASE)
            if match:
                scores[metric.lower().replace(" & ", "_").replace("-", "_").replace(" ", "_")] = int(match.group(1))
                break
    
    return scores

# Main evaluation function
def evaluate_models_automatic(models, test_questions, num_samples=200):
    results = {}
    
    # Limit to specified number of samples
    if len(test_questions) > num_samples:
        import random
        test_questions = random.sample(test_questions, num_samples)
    
    for model_name in models:
        print(f"Evaluating {model_name}...")
        model, tokenizer = models[model_name]
        
        gpt4_scores = []
        gemini_scores = []
        
        for question in tqdm(test_questions):
            # Generate response from model
            response = generate_response(model, tokenizer, question)
            
            # Evaluate with GPT-4
            gpt4_score = evaluate_with_gpt4(question, response)
            if gpt4_score:
                gpt4_scores.append(gpt4_score)
            
            # Evaluate with Gemini
            gemini_score = evaluate_with_gemini(question, response)
            if gemini_score:
                gemini_scores.append(gemini_score)
        
        # Calculate average scores
        gpt4_avg = {metric: np.mean([score.get(metric, 0) for score in gpt4_scores]) 
                   for metric in gpt4_scores[0].keys()}
        
        gemini_avg = {metric: np.mean([score.get(metric, 0) for score in gemini_scores]) 
                     for metric in gemini_scores[0].keys()}
        
        results[model_name] = {
            "gpt4": gpt4_avg,
            "gemini": gemini_avg,
            "combined": {metric: (gpt4_avg[metric] + gemini_avg[metric])/2 
                        for metric in gpt4_avg.keys()}
        }
        
        # Save intermediate results
        with open(f"evaluation_results_{model_name}.json", 'w') as f:
            json.dump(results, f, indent=2)
    
    return results

# Run evaluation (this will take time and API costs)
# test_questions = load_test_questions("test_questions.json")
# results = evaluate_models_automatic(models, test_questions, num_samples=200)