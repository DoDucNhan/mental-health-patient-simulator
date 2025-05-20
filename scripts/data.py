import argparse
import pandas as pd
from datasets import load_dataset
import random

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare MentalChat16K dataset for patient simulation")
    parser.add_argument(
        "--output_path",
        type=str,
        default="mental_health_patient_simulation_data.csv",
        help="Path to save the transformed dataset"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("Loading MentalChat16K dataset...")
    dataset = load_dataset("ShenLab/MentalChat16K")
    
    # Define patient simulation instruction
    patient_simulation_instruction = """You are simulating a patient with mental health challenges. You should respond as if you are the patient seeking help or expressing your thoughts and feelings. Try to convey emotions consistent with someone experiencing mental health difficulties. Remember to stay in character as the patient throughout the conversation."""
    
    print("Transforming dataset for patient simulation...")
    # Function to transform the dataset
    def transform_for_patient_simulation(example):
        # Swap input and output
        transformed_example = {
            "instruction": patient_simulation_instruction,
            "input": example["output"],  # The therapist's response becomes the input
            "output": example["input"]   # The patient's query becomes the output
        }
        return transformed_example
    
    # Transform the dataset
    transformed_train = dataset["train"].map(transform_for_patient_simulation)
    
    # Save the transformed dataset
    print(f"Saving transformed dataset to {args.output_path}...")
    transformed_train.to_pandas().to_csv(args.output_path, index=False)
    
    print("Dataset transformed and saved successfully!")
    print(f"Number of examples: {len(transformed_train)}")
    
    # Print a sample to verify
    print("\nSample example:")
    sample = transformed_train[0]
    print(f"Instruction: {sample['instruction'][:100]}...")
    print(f"Input (Therapist): {sample['input'][:100]}...")
    print(f"Output (Patient): {sample['output'][:100]}...")

if __name__ == "__main__":
    main()