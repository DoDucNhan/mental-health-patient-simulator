#!/bin/bash

# Exit on error
set -e

# Parse command line arguments
FORCE_ENV_CREATE=0
RUN_COMPARISON=1
SKIP_TRAINING=0
HF_TOKEN=""
WANDB_API_KEY=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --force-env-create)
      FORCE_ENV_CREATE=1
      shift
      ;;
    --skip-comparison)
      RUN_COMPARISON=0
      shift
      ;;
    --skip-training)
      SKIP_TRAINING=1
      shift
      ;;
    --hf-token)
      HF_TOKEN="$2"
      shift 2
      ;;
    --wandb-api-key)
      WANDB_API_KEY="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--force-env-create] [--skip-comparison] [--skip-training] [--hf-token TOKEN] [--wandb-api-key KEY]"
      exit 1
      ;;
  esac
done

# Set HF token if provided
if [ -n "$HF_TOKEN" ]; then
  export HF_TOKEN="$HF_TOKEN"
  export HUGGINGFACE_TOKEN="$HF_TOKEN"
  echo "Hugging Face token set from command line argument"
elif [ -n "$HUGGINGFACE_TOKEN" ]; then
  export HF_TOKEN="$HUGGINGFACE_TOKEN"
  echo "Hugging Face token set from environment variable HUGGINGFACE_TOKEN"
elif [ -z "$HF_TOKEN" ]; then
  read -sp "Enter your Hugging Face token (or leave empty if not needed): " HF_TOKEN
  echo
  if [ -n "$HF_TOKEN" ]; then
    export HF_TOKEN="$HF_TOKEN"
    export HUGGINGFACE_TOKEN="$HF_TOKEN"
    echo "Hugging Face token set from user input"
  fi
fi

# Set W&B API key if provided
if [ -n "$WANDB_API_KEY" ]; then
  export WANDB_API_KEY="$WANDB_API_KEY"
  echo "W&B API key set from command line argument"
elif [ -z "$WANDB_API_KEY" ]; then
  read -sp "Enter your Weights & Biases API key (or leave empty to disable W&B): " WANDB_API_KEY
  echo
  if [ -n "$WANDB_API_KEY" ]; then
    export WANDB_API_KEY="$WANDB_API_KEY"
    echo "W&B API key set from user input"
  fi
fi

# Create conda environment if it doesn't exist or if forced
if [ $FORCE_ENV_CREATE -eq 1 ] || ! conda info --envs | grep -q "patient_sim"; then
    echo "Creating conda environment 'patient_sim'..."
    conda create -n patient_sim python=3.10 -y
fi

# Activate environment
eval "$(conda shell.bash hook)"
conda activate patient_sim

# Install required packages
echo "Installing required packages..."
pip install torch==2.0.1 transformers==4.36.2 datasets peft bitsandbytes accelerate wandb tqdm

# Prepare the dataset
echo "Preparing dataset..."
python prepare_dataset.py

# Create models info JSON file with updated paths
cat > models_info.json << EOL
[
    {
        "name": "OpenHermes 2.5 Mistral",
        "base_model": "teknium/OpenHermes-2.5-Mistral-7B",
        "adapter_path": "./patient_simulation_OpenHermes-2.5-Mistral-7B/adapter"
    },
    {
        "name": "Phi-3 mini",
        "base_model": "microsoft/phi-3-mini-4k-instruct",
        "adapter_path": "./patient_simulation_phi-3-mini-4k-instruct/adapter"
    },
    {
        "name": "Gemma 7B Instruct",
        "base_model": "google/gemma-7b-it",
        "adapter_path": "./patient_simulation_gemma-7b-it/adapter"
    },
    {
        "name": "Llama 3 8B Instruct",
        "base_model": "meta-llama/Llama-3-8b-instruct",
        "adapter_path": "./patient_simulation_Llama-3-8b-instruct/adapter"
    }
]
EOL

if [ $SKIP_TRAINING -eq 0 ]; then
    # Models to fine-tune
    declare -a models=(
        "teknium/OpenHermes-2.5-Mistral-7B" 
        "microsoft/phi-3-mini-4k-instruct"
        "google/gemma-7b-it"
        "meta-llama/Llama-3-8b-instruct"
    )

    # Fine-tune each model
    for model in "${models[@]}"
    do
        echo "Starting fine-tuning for $model"
        
        # Extract model name for output directory
        model_name=$(echo "$model" | sed 's/.*\///')
        output_dir="./patient_simulation_${model_name}"
        
        # Run fine-tuning with authentication
        python finetune.py \
            --model_name "$model" \
            --output_dir "$output_dir" \
            --use_wandb
        
        echo "Completed fine-tuning for $model"
    done
else
    echo "Skipping training as requested. Using existing fine-tuned models."
fi

# Run comparison if enabled
if [ $RUN_COMPARISON -eq 1 ]; then
    echo "Running model comparison..."
    python compare_models.py --models_info models_info.json
    echo "Comparison completed!"
fi

echo "All processes completed successfully!"
echo ""
echo "To interact with a patient simulation model, use:"
echo "python interactive.py --model_name MODEL_NAME"
echo ""
echo "Examples:"
echo "python interactive.py --model_name patient_simulation_OpenHermes-2.5-Mistral-7B --style verbose"
echo "python interactive.py --model_name patient_simulation_phi-3-mini-4k-instruct --style reserved"