# Mental Health Patient Simulator

A comprehensive toolkit for training conversational AI agents to simulate patients with depression and anxiety, designed to help train mental health professionals and students.

## Overview

This project provides tools to fine-tune large language models (LLMs) to act as realistic patients with mental health conditions. The simulated patients can help medical students and mental health professionals practice therapeutic skills in a safe, controlled environment.

### Key Features

- **Fine-tuning Pipeline**: Complete workflow for training LLMs on mental health patient simulation data
- **Multiple Evaluation Methods**: Rule-based, LLM-based, and ensemble evaluation approaches
- **Interactive Applications**: Streamlit apps for expert evaluation and trainee training
- **Multiple Conversation Styles**: Plain, reserved, verbose, upset, tangent, and pleasing patient personas
- **Support for Multiple Models**: Compatible with popular open-source models (Llama, Mistral, Phi, Gemma)

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for model training)
- Sufficient disk space (models can be 10-20GB each)

### Setup

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd mental-health-patient-simulator
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up authentication** (optional, for some models)
   ```bash
   export HF_TOKEN="your_huggingface_token"
   ```

## Quick Start

### 1. Download Pre-trained Models

```bash
python scripts/download_models.py --models openhermes phi3 --output_dir ./models
```

Available models:

- `openhermes`: OpenHermes-2.5-Mistral-7B (no auth required)
- `phi3`: Phi-3-mini-4k-instruct (no auth required)
- `gemma`: Gemma-7B-Instruct (requires HF auth)
- `llama3`: Llama-3-8B-Instruct (requires HF auth + approval)

### 2. Prepare Training Data

```bash
python scripts/data.py --output_path mental_health_patient_simulation_data.csv
```

This downloads and transforms the MentalChat16K dataset for patient simulation training.

### 3. Fine-tune a Model

```bash
python scripts/finetune.py \
  --model_name teknium/OpenHermes-2.5-Mistral-7B \
  --dataset_path mental_health_patient_simulation_data.csv \
  --num_epochs 3 \
  --batch_size 4
```

### 4. Test Your Model

```bash
python scripts/interactive.py \
  --model_name patient_simulation_OpenHermes-2.5-Mistral-7B_* \
  --style plain
```

## Detailed Usage

### Training Pipeline

#### 1. Data Preparation

The training data is automatically downloaded from the MentalChat16K dataset:

```bash
python scripts/data.py --output_path custom_data.csv
```

#### 2. Model Fine-tuning

Fine-tune models with QLoRA for efficient training:

```bash
python scripts/finetune.py \
  --model_name microsoft/phi-3-mini-4k-instruct \
  --dataset_path mental_health_patient_simulation_data.csv \
  --num_epochs 5 \
  --batch_size 8 \
  --learning_rate 2e-4 \
  --lora_r 64 \
  --max_seq_length 1024
```

**Key Parameters:**

- `--model_name`: Base model from HuggingFace
- `--num_epochs`: Training epochs (3-5 recommended)
- `--batch_size`: Batch size per GPU
- `--lora_r`: LoRA rank (32-64 recommended)

### Evaluation

#### 1. Quick Evaluation

Simple rule-based evaluation without external dependencies:

```bash
python eval/simple_evaluation.py \
  --model_paths "MyModel:./patient_simulation_model" \
  --condition_type depression \
  --num_questions 50
```

#### 2. Comprehensive Evaluation

Multi-method evaluation using ensemble approach:

```bash
python eval/evaluate_models.py \
  --model_paths "Model1:/path/to/model1" "Model2:/path/to/model2" \
  --evaluator ensemble \
  --num_questions 100
```

#### 3. Expert Evaluation (Streamlit App)

Launch interactive evaluation interface for mental health professionals:

```bash
streamlit run eval/expert_eval_app.py
```

#### 4. Trainee Training (Streamlit App)

Launch training interface for students:

```bash
streamlit run eval/trainee_eval_app.py
```

### Interactive Testing

Test your trained models interactively:

```bash
python scripts/interactive.py \
  --model_name ./patient_simulation_model \
  --style reserved \
  --temperature 0.7
```

**Conversation Styles:**

- `plain`: Direct, straightforward responses
- `reserved`: Brief, hesitant responses
- `verbose`: Detailed, elaborate responses
- `upset`: Frustrated, resistant responses
- `tangent`: Off-topic, unfocused responses
- `pleasing`: Agreeable, approval-seeking responses

### Model Comparison

Compare multiple models side-by-side:

```bash
python scripts/compare.py \
  --models_info models_config.json \
  --num_test_prompts 10
```

## Configuration

### Environment Variables

```bash
export HF_TOKEN="your_huggingface_token"        # For accessing gated models
export CUDA_VISIBLE_DEVICES="0"                 # GPU selection
export TRANSFORMERS_CACHE="./cache"             # Model cache directory
```

### Model Configuration

Create a `models_config.json` file:

```json
[
  {
    "name": "Fine-tuned OpenHermes",
    "base_model": "teknium/OpenHermes-2.5-Mistral-7B",
    "adapter_path": "./patient_simulation_openhermes/adapter"
  },
  {
    "name": "Fine-tuned Phi3",
    "base_model": "microsoft/phi-3-mini-4k-instruct",
    "adapter_path": "./patient_simulation_phi3/adapter"
  }
]
```

## Evaluation Metrics

The system evaluates models across multiple dimensions:

### Automatic Metrics

- **Symptom Relevance**: Accuracy of symptom presentation
- **Personal Expression**: Use of first-person language
- **Length Appropriateness**: Response length naturalness
- **Question Relevance**: Appropriateness to therapist input
- **Emotional Expression**: Emotional language usage

### Expert Evaluation Metrics

- **Active Listening**: Patient's demonstration of understanding
- **Emotional Expression**: Authenticity of emotional expression
- **Clinical Realism**: Realistic symptom presentation
- **Conversational Quality**: Natural conversation flow
- **Educational Value**: Usefulness for training purposes

## Troubleshooting

### Common Issues

1. **Model Loading Errors**

   ```bash
   # Ensure you have sufficient GPU memory
   export CUDA_VISIBLE_DEVICES="0"
   # Or use CPU-only mode
   export CUDA_VISIBLE_DEVICES=""
   ```

2. **Authentication Issues**

   ```bash
   # For Llama/Gemma models
   huggingface-cli login
   # Or set token directly
   export HF_TOKEN="your_token"
   ```

3. **Memory Issues During Training**

   ```bash
   # Reduce batch size and enable gradient checkpointing
   python scripts/finetune.py \
     --batch_size 2 \
     --gradient_accumulation_steps 8
   ```

4. **Attention Mask Warnings**
   These are normal for some models and can be safely ignored.

### Performance Optimization

- **Use QLoRA**: Automatically enabled for 4-bit quantization
- **Gradient Checkpointing**: Enabled by default to save memory
- **Model Parallelism**: Automatically distributed across available GPUs

## File Structure

```
mental-health-patient-simulator/
├── scripts/                    # Core training and utility scripts
│   ├── data.py                # Data preparation
│   ├── finetune.py           # Model fine-tuning
│   ├── interactive.py        # Interactive testing
│   ├── compare.py            # Model comparison
│   └── download_models.py    # Model downloading
├── eval/                      # Evaluation tools and apps
│   ├── simple_evaluation.py  # Quick evaluation
│   ├── expert_eval_app.py    # Expert evaluation interface
│   ├── trainee_eval_app.py   # Trainee training interface
│   └── *_evaluator.py        # Various evaluation methods
├── models/                    # Downloaded models directory
├── results/                   # Evaluation results
└── requirements.txt          # Python dependencies
```

## Citation

This project builds upon the Patient-Ψ framework and MentalChat16K dataset. If you use this code, please cite:

```bibtex
@inproceedings{wang-etal-2024-patient,
    title = "{PATIENT}-$\psi$: Using Large Language Models to Simulate Patients for Training Mental Health Professionals",
    author = "Wang, Ruiyi  and
      Milani, Stephanie  and
      Chiu, Jamie C.  and
      Zhi, Jiayin  and
      Eack, Shaun M.  and
      Labrum, Travis  and
      Murphy, Samuel M  and
      Jones, Nev  and
      Hardy, Kate V  and
      Shen, Hong  and
      Fang, Fei  and
      Chen, Zhiyu",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.711",
    pages = "12772--12797",
}

@article{MentalChat16K,
  author    = {Jia Xu, Tianyi Wei, Bojian Hou, Patryk Orzechowski, Shu Yang, Ruochen Jin, Rachael Paulbeck, Joost Wagenaar, George Demiris, Li Shen},
  title     = {MentalChat16K: A Benchmark Dataset for Conversational Mental Health Assistance},
  year      = {2024},
  url       = {https://huggingface.co/datasets/ShenLab/MentalChat16K},
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## Support

For questions and support:

1. Check the troubleshooting section above
2. Open an issue on GitHub
3. Review the example scripts in the repository

---

**Note**: This tool is designed for educational and training purposes only. It should not be used as a substitute for real patient interactions or clinical assessment tools.
