# evaluate_models.py
import argparse
import json
import os
import torch
from typing import Dict, List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# Import the evaluator classes
from llama_evaluator import LlamaEvaluator
from mistral_evaluator import MistralEvaluator
from rule_based_evaluator import RuleBasedEvaluator
from ensemble_evaluator import EnsembleEvaluator

def load_model(model_path):
    """Load a model and tokenizer from the given path"""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return model, tokenizer

def generate_patient_response(model, tokenizer, question, condition_type="depression", conversation_style="plain"):
    """Generate a patient response using the model"""
    # Import instructions
    from mental_health_patient_instructions import get_depression_instructions, get_anxiety_instructions
    
    if condition_type == "depression":
        instructions = get_depression_instructions(conversation_style)
    else:
        instructions = get_anxiety_instructions(conversation_style)
    
    # Create the full prompt
    prompt = instructions.replace("[Question]", question).replace("Therapist: [Question]", f"Therapist: {question}")
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=1024,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the patient response part
    response = response[len(prompt):].strip()
    
    return response

def main():
    parser = argparse.ArgumentParser(description="Free evaluation of mental health patient simulation models")
    parser.add_argument("--model_paths", type=str, nargs="+", required=True)
    parser.add_argument("--evaluator", type=str, choices=["llama", "mistral", "rule_based", "ensemble"], 
                       default="ensemble", help="Choose evaluation method")
    parser.add_argument("--num_questions", type=int, default=100)
    parser.add_argument("--condition_type", type=str, choices=["depression", "anxiety"], default="depression")
    parser.add_argument("--output_dir", type=str, default="results")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load test questions
    with open("data/test_questions.json", "r") as f:
        test_questions = json.load(f)[:args.num_questions]
    
    # Initialize evaluator
    print(f"Initializing {args.evaluator} evaluator...")
    if args.evaluator == "llama":
        evaluator = LlamaEvaluator()
    elif args.evaluator == "mistral":
        evaluator = MistralEvaluator()
    elif args.evaluator == "rule_based":
        evaluator = RuleBasedEvaluator()
    else:
        evaluator = EnsembleEvaluator()
    
    # Load models and generate responses
    results = {}
    
    for model_path in args.model_paths:
        model_name, path = model_path.split(":", 1)
        print(f"Evaluating {model_name}...")
        
        # Load model
        model, tokenizer = load_model(path)
        
        # Generate responses and evaluate
        model_scores = []
        
        for question in tqdm(test_questions, desc=f"Evaluating {model_name}"):
            # Generate response
            response = generate_patient_response(model, tokenizer, question, args.condition_type)
            
            # Evaluate response
            if args.evaluator == "rule_based":
                scores = evaluator.evaluate_response(question, response, args.condition_type)
            else:
                scores = evaluator.evaluate_response(question, response)
            
            model_scores.append(scores)
        
        # Calculate average scores
        avg_scores = {}
        for metric in model_scores[0].keys():
            avg_scores[metric] = sum(score[metric] for score in model_scores) / len(model_scores)
        
        results[model_name] = avg_scores
        
        # Clean up GPU memory
        del model
        torch.cuda.empty_cache()
    
    # Save results
    with open(f"{args.output_dir}/evaluation_results_free.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Create visualizations
    create_visualizations(results, args.output_dir)
    
    # Print summary
    print_summary(results)

def create_visualizations(results: Dict, output_dir: str):
    """Create visualizations for free evaluation results"""
    df = pd.DataFrame(results).T
    
    # Bar chart comparing models
    plt.figure(figsize=(12, 8))
    df.plot(kind='bar', figsize=(12, 8))
    plt.title("Model Evaluation Scores (Free Evaluation)")
    plt.xlabel("Models")
    plt.ylabel("Scores (1-10)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/model_comparison_free.png", dpi=300)
    plt.close()
    
    # Heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(df, annot=True, cmap="YlOrRd", vmin=0, vmax=10)
    plt.title("Model Evaluation Heatmap (Free Evaluation)")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/evaluation_heatmap_free.png", dpi=300)
    plt.close()

def print_summary(results: Dict):
    """Print evaluation summary"""
    print("\n" + "="*50)
    print("FREE EVALUATION SUMMARY")
    print("="*50)
    
    # Find best performing model
    overall_scores = {model: scores.get('overall_quality', scores.get('overall', 0)) 
                     for model, scores in results.items()}
    best_model = max(overall_scores, key=overall_scores.get)
    
    print(f"Best performing model: {best_model}")
    print(f"Score: {overall_scores[best_model]:.2f}/10")
    
    print("\nAll model scores:")
    for model, score in sorted(overall_scores.items(), key=lambda x: x[1], reverse=True):
        print(f"  {model}: {score:.2f}/10")

if __name__ == "__main__":
    main()