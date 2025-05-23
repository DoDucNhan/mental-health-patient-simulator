# simple_evaluation.py
import json
import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings

# Suppress the specific warning
warnings.filterwarnings("ignore", message=".*attention_mask.*")

def simple_rule_based_eval(question, response, condition_type="depression"):
    """Simple rule-based evaluation without external dependencies"""
    
    # Define key indicators
    if condition_type == "depression":
        key_words = ['sad', 'hopeless', 'tired', 'worthless', 'guilty', 'empty', 'depressed', 'sleep', 'appetite']
    else:
        key_words = ['anxious', 'worried', 'panic', 'nervous', 'restless', 'overwhelmed', 'racing', 'tense']
    
    response_lower = response.lower()
    question_lower = question.lower()
    
    # Calculate scores
    scores = {}
    
    # 1. Symptom relevance (0-10)
    symptom_score = sum(1 for word in key_words if word in response_lower)
    scores['symptom_relevance'] = min(symptom_score * 2, 10)
    
    # 2. Personal expression (0-10) - first person usage
    first_person = ['i', 'me', 'my', 'myself', 'mine']
    personal_score = sum(1 for word in first_person if word in response_lower.split())
    scores['personal_expression'] = min(personal_score * 2, 10)
    
    # 3. Response length appropriateness (0-10)
    word_count = len(response.split())
    if 20 <= word_count <= 100:
        scores['length_appropriateness'] = 10
    elif 10 <= word_count < 20 or 100 < word_count <= 150:
        scores['length_appropriateness'] = 7
    else:
        scores['length_appropriateness'] = 4
    
    # 4. Question relevance (0-10)
    question_words = set(question_lower.split()) - {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    response_words = set(response_lower.split())
    overlap = len(question_words.intersection(response_words))
    scores['question_relevance'] = min(overlap * 2, 10)
    
    # 5. Emotional expression (0-10)
    emotion_words = ['feel', 'feeling', 'felt', 'emotion', 'mood']
    emotion_score = sum(1 for word in emotion_words if word in response_lower)
    scores['emotional_expression'] = min(emotion_score * 3, 10)
    
    # 6. Overall score
    scores['overall'] = np.mean(list(scores.values()))
    
    return scores

def load_model_fixed(model_path):
    """Load model and tokenizer with proper attention mask handling"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Fix tokenizer padding issues
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        # Ensure pad_token_id is different from eos_token_id if possible
        if tokenizer.pad_token_id == tokenizer.eos_token_id and hasattr(tokenizer, 'unk_token_id'):
            if tokenizer.unk_token_id is not None and tokenizer.unk_token_id != tokenizer.eos_token_id:
                tokenizer.pad_token_id = tokenizer.unk_token_id
                tokenizer.pad_token = tokenizer.unk_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Resize embeddings if we added tokens
        model.resize_token_embeddings(len(tokenizer))
        
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None, None

def generate_response_fixed(model, tokenizer, question, condition_type="depression"):
    """Generate patient response with proper attention mask handling"""
    
    if condition_type == "depression":
        system_prompt = """You are simulating a patient with depression. Respond naturally to the therapist's question. Show symptoms like sadness, hopelessness, fatigue, and negative thinking patterns."""
    else:
        system_prompt = """You are simulating a patient with anxiety. Respond naturally to the therapist's question. Show symptoms like worry, restlessness, panic, and catastrophic thinking."""
    
    prompt = f"{system_prompt}\n\nTherapist: {question}\nPatient:"
    
    # Tokenize with proper attention mask
    inputs = tokenizer(
        prompt, 
        return_tensors="pt", 
        padding=True,
        truncation=True,
        max_length=512,
        return_attention_mask=True
    ).to(model.device)
    
    # Generate with attention mask
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,  # Explicitly pass attention mask
            max_length=768,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,  # Prevent repetition
            repetition_penalty=1.1
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response[len(prompt):].strip()
    
    # Clean up response
    if response.startswith("Patient:"):
        response = response[8:].strip()
    
    return response

def main():
    parser = argparse.ArgumentParser(description="Simple free evaluation with fixed attention masks")
    parser.add_argument("--model_paths", type=str, nargs="+", required=True)
    parser.add_argument("--questions_file", type=str, default="test_questions.json")
    parser.add_argument("--condition_type", type=str, choices=["depression", "anxiety"], default="depression")
    parser.add_argument("--num_questions", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="results")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load questions
    try:
        with open(args.questions_file, "r") as f:
            questions = json.load(f)[:args.num_questions]
    except FileNotFoundError:
        print(f"Questions file not found: {args.questions_file}")
        print("Please run: python src/create_test_questions.py")
        return
    
    results = {}
    
    for model_path in args.model_paths:
        model_name, path = model_path.split(":", 1)
        print(f"Evaluating {model_name}...")
        
        # Load model
        model, tokenizer = load_model_fixed(path)
        
        if model is None or tokenizer is None:
            print(f"Failed to load {model_name}, skipping...")
            continue
        
        model_scores = []
        
        for question in tqdm(questions, desc=f"Processing {model_name}"):
            try:
                # Generate response
                response = generate_response_fixed(model, tokenizer, question, args.condition_type)
                
                # Evaluate
                scores = simple_rule_based_eval(question, response, args.condition_type)
                model_scores.append(scores)
            except Exception as e:
                print(f"Error processing question for {model_name}: {e}")
                # Add default scores for failed generations
                default_scores = {
                    'symptom_relevance': 1,
                    'personal_expression': 1,
                    'length_appropriateness': 1,
                    'question_relevance': 1,
                    'emotional_expression': 1,
                    'overall': 1
                }
                model_scores.append(default_scores)
        
        # Calculate averages
        if model_scores:
            avg_scores = {}
            for metric in model_scores[0].keys():
                avg_scores[metric] = np.mean([score[metric] for score in model_scores])
            
            results[model_name] = avg_scores
        
        # Clean up
        del model, tokenizer
        torch.cuda.empty_cache()
    
    if not results:
        print("No models were successfully evaluated!")
        return
    
    # Save results
    with open(f"{args.output_dir}/simple_evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Print results
    print("\nEvaluation Results:")
    print("="*50)
    
    df = pd.DataFrame(results).T
    print(df.round(2))
    
    # Find best model
    best_model = df['overall'].idxmax()
    print(f"\nBest performing model: {best_model}")
    print(f"Overall score: {df.loc[best_model, 'overall']:.2f}/10")

if __name__ == "__main__":
    main()