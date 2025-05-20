import os
import torch
import argparse
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
import wandb
from datetime import datetime
import getpass
import logging
import warnings

# Setup logging and suppress warnings
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore", message=".*The current process just got forked.*")
warnings.filterwarnings("ignore", category=UserWarning)

# Add TikToken workaround
try:
    import tiktoken
    # Force tiktoken to use the right encoding for certain models
    # This helps avoid the "not enough values to unpack" error
    os.environ["TIKTOKEN_CACHE_DIR"] = ".tiktoken_cache"
except ImportError:
    pass

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune LLMs for mental health patient simulation")
    parser.add_argument(
        "--model_name",
        type=str,
        default="teknium/OpenHermes-2.5-Mistral-7B",
        help="Model identifier from Hugging Face Hub"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="mental_health_patient_simulation_data.csv",
        help="Path to the prepared dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save the model (default: ./patient_simulation_{model_name}_{datetime})"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size per GPU"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of gradient accumulation steps"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="LoRA rank dimension"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha parameter"
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="LoRA dropout probability"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Maximum sequence length for training"
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Whether to use Weights & Biases for tracking"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="mental_health_patient_simulation",
        help="W&B project name"
    )
    parser.add_argument(
        "--wandb_api_key",
        type=str,
        default=None,
        help="Weights & Biases API key (if not provided, will check env var or prompt)"
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Hugging Face API token (if not provided, will check env var or prompt)"
    )
    
    args = parser.parse_args()
    
    # Set output directory if not specified
    if args.output_dir is None:
        model_name_short = args.model_name.split('/')[-1]
        args.output_dir = f"./patient_simulation_{model_name_short}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    
    return args

def setup_authentication(args):
    """Set up authentication for HuggingFace and W&B"""
    
    # Set up Hugging Face token
    hf_token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    
    if not hf_token:
        print("Hugging Face token not found in arguments or environment variables.")
        hf_token = getpass.getpass("Enter your Hugging Face token (or leave empty if not needed): ")
    
    if hf_token:
        os.environ["HUGGINGFACE_TOKEN"] = hf_token
        os.environ["HF_TOKEN"] = hf_token
        print("Hugging Face token set.")
    
    # Set up W&B if requested
    if args.use_wandb:
        wandb_key = args.wandb_api_key or os.environ.get("WANDB_API_KEY")
        
        if not wandb_key:
            print("W&B API key not found in arguments or environment variables.")
            wandb_key = getpass.getpass("Enter your Weights & Biases API key: ")
        
        if wandb_key:
            os.environ["WANDB_API_KEY"] = wandb_key
            print("W&B API key set.")
            
            # Initialize wandb
            wandb.login()
            wandb.init(project=args.wandb_project)
        else:
            print("No W&B API key provided. Disabling W&B integration.")
            args.use_wandb = False

def main():
    # Parse arguments
    args = parse_args()
    
    # Setup authentication
    setup_authentication(args)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # QLoRA Configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    try:
        print(f"Loading model: {args.model_name}")
        
        # Load the base model with quantization
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            token=os.environ.get("HF_TOKEN"),
            # use_fast_tokenizer=False 
        )
        
        # Load tokenizer with more robust settings
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name, 
            trust_remote_code=True,
            padding_side="right",
            token=os.environ.get("HF_TOKEN"),
            use_fast=False  # Add this line to avoid tiktoken issues
        )
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        raise
    
    # Make sure the tokenizer has padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Define LoRA configuration
    target_modules = None
    
    # Set appropriate target modules based on model architecture
    if "mistral" in args.model_name.lower() or "openhermes" in args.model_name.lower():
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    elif "llama" in args.model_name.lower():
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    elif "phi" in args.model_name.lower():
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    elif "gemma" in args.model_name.lower():
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    else:
        # Default target modules if model architecture not recognized
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Apply LoRA adapters to the model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # Print number of trainable parameters
    
    # Load the dataset
    print(f"Loading dataset from: {args.dataset_path}")
    dataset = load_dataset("csv", data_files=args.dataset_path)
    
    # Format the prompt template based on model
    def get_prompt_template(model_name):
        if "mistral" in model_name.lower() or "openhermes" in model_name.lower():
            return "<s>[INST] {instruction}\n\n{input} [/INST] {output} </s>"
        elif "llama" in model_name.lower():
            return "<s>[INST] {instruction}\n\n{input} [/INST] {output} </s>"
        elif "phi" in model_name.lower():
            return "<|user|>\n{instruction}\n\n{input}<|assistant|>\n{output}"
        elif "gemma" in model_name.lower():
            return "<start_of_turn>user\n{instruction}\n\n{input}<end_of_turn>\n<start_of_turn>model\n{output}<end_of_turn>"
        else:
            return "{instruction}\n\n{input}\n\n{output}"
    
    prompt_template = get_prompt_template(args.model_name)
    print(f"Using prompt template: {prompt_template}")
    
    # Tokenization function
    def tokenize_function(examples):
        try:
            # Combine instruction, input, and output into the appropriate format
            prompts = [
                prompt_template.format(
                    instruction=examples["instruction"][i],
                    input=examples["input"][i],
                    output=examples["output"][i]
                ) for i in range(len(examples["instruction"]))
            ]
            
            # Tokenize with padding
            tokenized_inputs = tokenizer(
                prompts, 
                padding="max_length",
                truncation=True,
                max_length=args.max_seq_length,
                return_tensors="pt"
            )
            
            # Create labels (same as input_ids for causal language modeling)
            tokenized_inputs["labels"] = tokenized_inputs["input_ids"].clone()
            
            # Apply loss masking: only calculate loss on the generated output tokens
            for i, prompt in enumerate(prompts):
                try:
                    # Find where the output starts in the prompt template
                    instruction_input = prompt_template.format(
                        instruction=examples["instruction"][i],
                        input=examples["input"][i],
                        output=""
                    )
                    
                    # Encode the instruction and input part - use a safer approach
                    instruction_input_tokens = tokenizer(
                        instruction_input, 
                        add_special_tokens=False,
                        return_tensors="pt"
                    )["input_ids"].shape[1]
                    
                    # Set instruction and input tokens to -100 to ignore in loss calculation
                    tokenized_inputs["labels"][i, :instruction_input_tokens] = -100
                except Exception as e:
                    logging.warning(f"Error masking labels for example {i}: {str(e)}")
                    # Fallback: Just keep all labels
            
            return tokenized_inputs
        except Exception as e:
            logging.error(f"Tokenization error: {str(e)}")
            # Return something valid to avoid breaking the pipeline
            return {
                "input_ids": torch.zeros((1, 10), dtype=torch.long),
                "attention_mask": torch.zeros((1, 10), dtype=torch.long),
                "labels": torch.zeros((1, 10), dtype=torch.long)
            }
    
    # Tokenize the dataset
    print("Tokenizing dataset...")
    tokenized_dataset = dataset["train"].map(
        tokenize_function,
        batched=True,
        remove_columns=["instruction", "input", "output"],
        desc="Tokenizing dataset"
    )
    
    # Create data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt"
    )
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=0.03,
        weight_decay=0.01,
        fp16=True,
        logging_steps=50,
        save_strategy="epoch",
        report_to="wandb" if args.use_wandb else "none",
        save_total_limit=3,
        push_to_hub=False,
        gradient_checkpointing=True,
    )
    
    # Create the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    # Train the model
    print(f"Starting training for {args.num_epochs} epochs...")
    trainer.train()
    
    # Save the final model
    print(f"Saving model to {args.output_dir}/final")
    trainer.save_model(f"{args.output_dir}/final")
    
    # Save the adapter separately for easy loading
    print(f"Saving adapter to {args.output_dir}/adapter")
    model.save_pretrained(f"{args.output_dir}/adapter")
    
    # Also save tokenizer
    tokenizer.save_pretrained(f"{args.output_dir}/final")
    
    print(f"Training completed successfully! Model saved to {args.output_dir}")
    
    # Finish W&B run if active
    if args.use_wandb and wandb.run is not None:
        wandb.finish()

if __name__ == "__main__":
    main()