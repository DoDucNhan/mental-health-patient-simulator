import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import sys
import os
import getpass
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Interactive patient simulation with fine-tuned LLMs")
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Name of the fine-tuned model directory (e.g., patient_simulation_OpenHermes-2.5-Mistral-7B)"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="Base model to use (if not auto-detected from model_name)"
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help="Path to the adapter weights (if not auto-detected from model_name)"
    )
    parser.add_argument(
        "--style",
        type=str,
        default="plain",
        choices=["plain", "upset", "verbose", "reserved", "tangent", "pleasing"],
        help="Conversational style"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for generation"
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load model in 8-bit precision (less VRAM, lower quality)"
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true", 
        help="Load model in 4-bit precision (even less VRAM, lower quality)"
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Hugging Face API token (if not provided, will check env var or prompt)"
    )
    parser.add_argument(
        "--models_info",
        type=str,
        default=None,
        help="JSON file containing information about available models"
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

def get_model_info(args):
    """Determine model information based on arguments or auto-detection"""
    # If all params are provided directly, use them
    if args.base_model and args.adapter_path:
        return {
            "name": args.model_name or "Unspecified model",
            "base_model": args.base_model,
            "adapter_path": args.adapter_path
        }
    
    # If models_info is provided, look up the model
    if args.models_info and os.path.exists(args.models_info):
        with open(args.models_info, 'r') as f:
            models_info = json.load(f)
            
        if args.model_name:
            # Find matching model
            for model in models_info:
                if args.model_name in model["name"]:
                    return model
    
    # Auto-detect from model_name
    if args.model_name:
        # Check if this is a directory name
        if os.path.isdir(args.model_name):
            dir_name = args.model_name
        else:
            dir_name = f"patient_simulation_{args.model_name}"
            if not os.path.isdir(dir_name):
                dir_name = args.model_name  # fallback
        
        # Try to determine base model from directory name
        if "openhermes" in dir_name.lower() or "mistral" in dir_name.lower():
            base_model = "teknium/OpenHermes-2.5-Mistral-7B"
        elif "phi" in dir_name.lower():
            base_model = "microsoft/phi-3-mini-4k-instruct"
        elif "gemma" in dir_name.lower():
            base_model = "google/gemma-7b-it"
        elif "llama" in dir_name.lower():
            base_model = "meta-llama/Llama-3-8b-instruct"
        else:
            raise ValueError(f"Cannot determine base model from {dir_name}. Please specify --base_model explicitly.")
        
        # Check for adapter path
        adapter_path = os.path.join(dir_name, "adapter")
        if not os.path.exists(adapter_path):
            raise ValueError(f"Adapter not found at {adapter_path}. Please specify --adapter_path explicitly.")
        
        return {
            "name": dir_name,
            "base_model": base_model,
            "adapter_path": adapter_path
        }
    
    # If we get here, we don't have enough information
    raise ValueError("Insufficient model information. Please specify either --model_name, or both --base_model and --adapter_path.")

def main():
    args = parse_args()
    
    # Setup authentication
    hf_token = setup_authentication(args)
    
    # Define style descriptions
    style_descriptions = {
        "plain": "direct, straightforward",
        "upset": "frustrated, resistant, challenging or dismissive of the therapist",
        "verbose": "providing detailed, extensive responses, even to simple questions",
        "reserved": "providing brief, vague, or evasive answers, requiring more prompting to open up",
        "tangent": "starting to answer but quickly veering off into unrelated topics",
        "pleasing": "eager-to-please, avoiding expressing disagreement, seeking approval"
    }
    
    try:
        # Get model information
        model_info = get_model_info(args)
        
        print(f"\nLoading fine-tuned patient simulation model:")
        print(f"Name: {model_info['name']}")
        print(f"Base model: {model_info['base_model']}")
        print(f"Adapter path: {model_info['adapter_path']}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_info['base_model'],
            trust_remote_code=True,
            token=hf_token
        )
        
        # Set up quantization config if needed
        model_kwargs = {
            "device_map": "auto",
            "trust_remote_code": True,
            "token": hf_token
        }
        
        if args.load_in_8bit:
            model_kwargs["load_in_8bit"] = True
        elif args.load_in_4bit:
            model_kwargs["load_in_4bit"] = True
        else:
            model_kwargs["torch_dtype"] = torch.float16
        
        # Load quantized base model
        base_model = AutoModelForCausalLM.from_pretrained(
            model_info['base_model'],
            **model_kwargs
        )
        
        # Load fine-tuned model with adapter
        if os.path.exists(model_info['adapter_path']):
            model = PeftModel.from_pretrained(base_model, model_info['adapter_path'])
            print(f"Successfully loaded adapter from {model_info['adapter_path']}")
        else:
            raise ValueError(f"Adapter not found at {model_info['adapter_path']}")
        
        model.eval()
        
        # Instruction with style - this is for PATIENT simulation
        instruction = f"You are simulating a patient with depression and anxiety. You should respond as the patient using a {args.style} conversational style, which means being {style_descriptions[args.style]}. Stay in character as the patient throughout the conversation."
        
        print(f"\nPatient Simulation ({args.style} style)")
        print(f"Type 'quit' to exit")
        print(f"You are the therapist, the AI is the patient.\n")
        
        conversation_history = []
        
        while True:
            therapist_input = input("\nTherapist: ")
            if therapist_input.lower() == 'quit':
                break
            
            # Add to conversation history
            conversation_history.append(f"Therapist: {therapist_input}")
            
            # Format the prompt with history context if available
            context = "\n".join(conversation_history[-5:])  # Last 5 exchanges for context
            formatted_prompt = get_formatted_prompt(model_info['base_model'], instruction, context)
            
            inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
            
            # Generate response
            with torch.no_grad():
                output = model.generate(
                    inputs["input_ids"],
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    do_sample=True
                )
            
            # Decode and clean response
            response = tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Extract just the generated part (remove the prompt)
            response = response.replace(formatted_prompt, "").strip()
            
            print(f"\nPatient: {response}")
            
            # Add to conversation history
            conversation_history.append(f"Patient: {response}")
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())