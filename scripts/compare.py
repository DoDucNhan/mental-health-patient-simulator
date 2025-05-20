import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import pandas as pd
import time
import gc
import os
import json
from tqdm import tqdm
import getpass

def parse_args():
    parser = argparse.ArgumentParser(description="Compare fine-tuned patient simulation models")
    parser.add_argument(
        "--models_info",
        type=str,
        default=None,
        help="JSON file containing information about models to compare (optional)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="patient_simulation_model_comparison.csv",
        help="Path to save comparison results"
    )
    parser.add_argument(
        "--num_test_prompts",
        type=int,
        default=5,
        help="Number of test prompts to use"
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Hugging Face API token (if not provided, will check env var or prompt)"
    )
    return parser.parse_args()

def setup_authentication(args):
    """Set up authentication for HuggingFace"""
    
    # Set up Hugging Face token
    hf_token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    
    if not hf_token:
        print("Hugging Face token not found in arguments or environment variables.")
        hf_token = getpass.getpass("Enter your Hugging Face token (or leave empty if not needed): ")
    
    if hf_token:
        os.environ["HUGGINGFACE_TOKEN"] = hf_token
        os.environ["HF_TOKEN"] = hf_token
        print("Hugging Face token set.")
    
    return hf_token

def get_formatted_prompt(model_name, instruction, input_text):
    if "mistral" in model_name.lower() or "openhermes" in model_name.lower():
        return f"<s>[INST] {instruction}\n\n{input_text} [/INST]"
    elif "llama" in model_name.lower():
        return f"<s>[INST] {instruction}\n\n{input_text} [/INST]"
    elif "phi" in model_name.lower():
        return f"<|user|>\n{instruction}\n\n{input_text}<|assistant|>\n"
    elif "gemma" in model_name.lower():
        return f"<start_of_turn>user\n{instruction}\n\n{input_text}<end_of_turn>\n<start_of_turn>model\n"
    else:
        return f"{instruction}\n\n{input_text}\n\n"

def main():
    args = parse_args()
    
    # Setup authentication
    hf_token = setup_authentication(args)
    
    # Auto-discover fine-tuned models if models_info not provided
    if args.models_info and os.path.exists(args.models_info):
        with open(args.models_info, 'r') as f:
            models_to_compare = json.load(f)
        print(f"Loaded {len(models_to_compare)} models from {args.models_info}")
    else:
        # Auto-discover model directories
        models_to_compare = []
        base_dir = "./"
        
        for dir_name in os.listdir(base_dir):
            if dir_name.startswith("patient_simulation_") and os.path.isdir(os.path.join(base_dir, dir_name)):
                adapter_path = os.path.join(base_dir, dir_name, "adapter")
                
                # Extract base model name from directory
                model_name_parts = dir_name.replace("patient_simulation_", "").split("_")
                
                # Try to determine base model
                if "openhermes" in dir_name.lower() or "mistral" in dir_name.lower():
                    base_model = "teknium/OpenHermes-2.5-Mistral-7B"
                elif "phi" in dir_name.lower():
                    base_model = "microsoft/phi-3-mini-4k-instruct"
                elif "gemma" in dir_name.lower():
                    base_model = "google/gemma-7b-it"
                elif "llama" in dir_name.lower():
                    base_model = "meta-llama/Llama-3-8b-instruct"
                else:
                    # Skip if base model can't be determined
                    print(f"Skipping directory {dir_name} - base model unknown")
                    continue
                
                if os.path.exists(adapter_path):
                    models_to_compare.append({
                        "name": f"Fine-tuned {dir_name}",
                        "base_model": base_model,
                        "adapter_path": adapter_path
                    })
        
        if not models_to_compare:
            print("No fine-tuned models found. Please train models first or provide a models_info file.")
            return 1
        
        print(f"Auto-discovered {len(models_to_compare)} fine-tuned models")
    
    # Test prompts specifically for therapist-to-patient interactions
    # This is key: the therapist is asking, patient (model) is responding!
    test_prompts = [
        {"instruction": "You are simulating a patient with depression and anxiety. Respond as if you are the patient seeking help.", 
         "input": "How have you been feeling lately? Have there been any changes in your mood or energy levels?"},
        {"instruction": "You are simulating a patient with depression. Respond as the patient would.", 
         "input": "Have you noticed any changes in your sleep patterns or appetite recently?"},
        {"instruction": "You are simulating a patient with anxiety. Respond authentically as if you are experiencing anxiety.", 
         "input": "What situations tend to make you feel more anxious or worried?"},
        {"instruction": "You are simulating a patient with mental health challenges using a reserved conversational style (minimal, restrained responses).", 
         "input": "I notice you've been quiet today. Is there something specific on your mind?"},
        {"instruction": "You are simulating a patient with mental health challenges using a verbose conversational style (detailed, expressive responses).", 
         "input": "Tell me about how your week has been going."}
    ]
    
    # Limit test prompts if specified
    test_prompts = test_prompts[:args.num_test_prompts]
    
    # Results storage
    results = []
    
    # Run comparison
    for model_info in models_to_compare:
        print(f"\nEvaluating {model_info['name']}...")
        
        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_info['base_model'],
                trust_remote_code=True,
                token=hf_token
            )
            
            # Load quantized base model
            base_model = AutoModelForCausalLM.from_pretrained(
                model_info['base_model'],
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
                token=hf_token
            )
            
            # Load fine-tuned model with adapter
            if os.path.exists(model_info['adapter_path']):
                model = PeftModel.from_pretrained(base_model, model_info['adapter_path'])
                print(f"Loaded adapter from {model_info['adapter_path']}")
            else:
                model = base_model
                print(f"Adapter path not found, using base model")
            
            model.eval()
            
            # Test each prompt
            for prompt_data in tqdm(test_prompts, desc=f"Testing {model_info['name']}"):
                formatted_prompt = get_formatted_prompt(
                    model_info['base_model'], 
                    prompt_data['instruction'], 
                    prompt_data['input']
                )
                
                inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
                
                # Measure memory usage before generation
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    start_mem = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
                
                # Measure inference time
                start_time = time.time()
                
                # Generate response
                with torch.no_grad():
                    output = model.generate(
                        inputs["input_ids"],
                        max_new_tokens=512,
                        temperature=0.7,
                        top_p=0.9,
                        repetition_penalty=1.1,
                        do_sample=True
                    )
                
                end_time = time.time()
                inference_time = end_time - start_time
                
                # Measure peak memory usage
                if torch.cuda.is_available():
                    peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
                    mem_used = peak_mem - start_mem
                else:
                    mem_used = -1
                
                # Decode and clean response
                response = tokenizer.decode(output[0], skip_special_tokens=True)
                
                # Extract just the generated part (remove the prompt)
                response = response.replace(formatted_prompt, "").strip()
                
                # Store result
                results.append({
                    "Model": model_info['name'],
                    "Base Model": model_info['base_model'],
                    "Adapter Path": model_info['adapter_path'],
                    "Instruction": prompt_data['instruction'],
                    "Therapist Input": prompt_data['input'],
                    "Patient Response": response,
                    "Inference Time (s)": inference_time,
                    "Response Length": len(response.split()),
                    "Peak Memory (MB)": mem_used if torch.cuda.is_available() else "N/A"
                })
            
            # Clear memory
            del model
            del base_model
            torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            print(f"Error evaluating {model_info['name']}: {str(e)}")
            # Add error entry to results
            for prompt_data in test_prompts:
                results.append({
                    "Model": model_info['name'],
                    "Base Model": model_info['base_model'],
                    "Adapter Path": model_info['adapter_path'],
                    "Instruction": prompt_data['instruction'],
                    "Therapist Input": prompt_data['input'],
                    "Patient Response": f"ERROR: {str(e)}",
                    "Inference Time (s)": -1,
                    "Response Length": 0,
                    "Peak Memory (MB)": "N/A"
                })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results to CSV
    results_df.to_csv(args.output_file, index=False)
    
    print(f"\nComparison completed and saved to {args.output_file}")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    summary = results_df.groupby("Model").agg({
        "Inference Time (s)": ["mean", "std"],
        "Response Length": ["mean", "std"]
    })
    print(summary)
    
    return 0

if __name__ == "__main__":
    main()