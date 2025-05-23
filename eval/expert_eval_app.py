# expert_evaluation_app.py
import streamlit as st
import pandas as pd
import json
import random
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mental_health_patient_instructions import get_depression_instructions, get_anxiety_instructions

def setup_page():
    st.set_page_config(
        page_title="Mental Health Patient Simulation Evaluation",
        page_icon="🧠",
        layout="wide"
    )
    
    st.title("Mental Health Patient Simulation Evaluation")
    st.write("""
    This application allows mental health professionals to evaluate the quality and realism 
    of AI-simulated patient responses in a therapeutic context. Your evaluations will help 
    improve the training of these models for educational purposes.
    """)

@st.cache_resource
def load_model(model_path):
    """Load and cache model to avoid reloading"""
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model from {model_path}: {e}")
        return None, None

def generate_patient_response(model, tokenizer, question, condition_type="depression", conversation_style="plain"):
    """Generate patient response using the loaded model"""
    if model is None or tokenizer is None:
        return "Error: Model not loaded properly"
    
    try:
        # Get appropriate instructions
        if condition_type == "depression":
            instructions = get_depression_instructions(conversation_style)
        else:
            instructions = get_anxiety_instructions(conversation_style)
        
        # Create the full prompt
        prompt = instructions.replace("[Question]", question).replace("Therapist: [Question]", f"Therapist: {question}")
        
        inputs = tokenizer(prompt, 
                           return_tensors="pt", 
                           truncation=True, 
                           max_length=1024, 
                           return_attention_mask=True
                        ).to(model.device)
        
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
    except Exception as e:
        return f"Error generating response: {e}"

def load_or_generate_responses():
    """Load existing responses or generate new ones"""
    # Check if we have pre-generated responses
    responses_dir = "responses"
    if os.path.exists(responses_dir):
        response_files = [f for f in os.listdir(responses_dir) if f.endswith(".json")]
        if response_files:
            st.sidebar.header("Response Source")
            use_existing = st.sidebar.radio(
                "Choose response source:",
                ["Generate new responses", "Use existing responses"]
            )
            
            if use_existing == "Use existing responses":
                selected_file = st.sidebar.selectbox("Select response file:", response_files)
                with open(os.path.join(responses_dir, selected_file), "r") as f:
                    return json.load(f), "existing"
    
    # Generate new responses
    return generate_new_responses(), "new"

def generate_new_responses():
    """Generate new responses from models"""
    st.sidebar.header("Model Configuration")
    
    # Model selection
    model_paths = st.sidebar.text_area(
        "Model paths (one per line, format: name:path):",
        value="Phi3:/path/to/phi3\nGemma:/path/to/gemma",
        help="Enter model paths in format 'ModelName:/path/to/model'"
    ).strip().split('\n')
    
    model_paths = [path.strip() for path in model_paths if path.strip()]
    
    if not model_paths:
        st.error("Please enter at least one model path")
        return {}
    
    # Parse model paths
    models_config = {}
    for model_path in model_paths:
        if ':' in model_path:
            name, path = model_path.split(':', 1)
            models_config[name.strip()] = path.strip()
        else:
            st.error(f"Invalid format for model path: {model_path}")
            return {}
    
    # Session configuration
    condition_type = st.sidebar.selectbox("Condition Type:", ["depression", "anxiety"])
    conversation_style = st.sidebar.selectbox(
        "Conversation Style:", 
        ["plain", "reserved", "verbose", "upset", "tangent", "pleasing"]
    )
    num_questions = st.sidebar.slider("Number of questions:", 5, 20, 10)
    
    if st.sidebar.button("Generate Responses"):
        # Load test questions
        try:
            with open("data/test_questions.json", "r") as f:
                all_questions = json.load(f)
        except FileNotFoundError:
            st.error("Test questions not found. Please run create_test_questions.py first.")
            return {}
        
        # Select random questions
        selected_questions = random.sample(all_questions, min(num_questions, len(all_questions)))
        
        responses = {}
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_work = len(selected_questions) * len(models_config)
        current_work = 0
        
        for question in selected_questions:
            responses[question] = {}
            
            for model_name, model_path in models_config.items():
                status_text.text(f"Generating response from {model_name}...")
                
                # Load model
                model, tokenizer = load_model(model_path)
                
                if model is not None and tokenizer is not None:
                    # Generate response
                    response = generate_patient_response(
                        model, tokenizer, question, condition_type, conversation_style
                    )
                    responses[question][model_name] = response
                else:
                    responses[question][model_name] = "Error: Could not load model"
                
                current_work += 1
                progress_bar.progress(current_work / total_work)
        
        # Save responses
        os.makedirs("responses", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"responses/expert_eval_responses_{timestamp}.json"
        
        with open(filename, "w") as f:
            json.dump(responses, f, indent=2)
        
        status_text.text(f"Responses saved to {filename}")
        progress_bar.empty()
        
        return responses
    
    return {}

def expert_information():
    st.sidebar.header("Evaluator Information")
    
    expert_name = st.sidebar.text_input("Your Name")
    
    expert_credentials = st.sidebar.selectbox(
        "Professional Credentials", 
        ["Psychiatrist", "Psychologist", "Licensed Therapist", "Mental Health Counselor", 
         "Psychiatric Nurse", "Social Worker", "Other"]
    )
    
    if expert_credentials == "Other":
        other_credentials = st.sidebar.text_input("Please specify:")
        if other_credentials:
            expert_credentials = other_credentials
    
    years_experience = st.sidebar.number_input(
        "Years of Experience", 
        min_value=0, 
        max_value=50, 
        value=5
    )
    
    depression_experience = st.sidebar.slider(
        "Experience with Depression Patients (1-10)", 
        1, 10, 5
    )
    
    anxiety_experience = st.sidebar.slider(
        "Experience with Anxiety Patients (1-10)", 
        1, 10, 5
    )
    
    return {
        "name": expert_name,
        "credentials": expert_credentials,
        "years_experience": years_experience,
        "depression_experience": depression_experience,
        "anxiety_experience": anxiety_experience
    }

def prepare_evaluation_session(responses):
    if "evaluation_state" not in st.session_state:
        # Initialize evaluation state
        st.session_state.evaluation_state = {
            "questions": list(responses.keys()),
            "current_index": 0,
            "selected_questions": [],
            "model_mapping": {},
            "evaluations": {},
            "completed": False
        }
        
        # Select a subset of questions randomly
        num_questions = min(10, len(st.session_state.evaluation_state["questions"]))
        st.session_state.evaluation_state["selected_questions"] = random.sample(
            st.session_state.evaluation_state["questions"], 
            num_questions
        )
        
        # For each question, randomly assign models to letters
        for question in st.session_state.evaluation_state["selected_questions"]:
            models = list(responses[question].keys())
            random.shuffle(models)
            st.session_state.evaluation_state["model_mapping"][question] = {
                letter: model for letter, model in zip(["A", "B", "C", "D"], models[:4])
            }
    
    return st.session_state.evaluation_state

def evaluate_responses(responses, evaluation_state, expert_info):
    # Get current question
    if evaluation_state["current_index"] >= len(evaluation_state["selected_questions"]):
        evaluation_state["completed"] = True
        return
    
    current_question = evaluation_state["selected_questions"][evaluation_state["current_index"]]
    
    # Display progress
    st.progress((evaluation_state["current_index"]) / len(evaluation_state["selected_questions"]))
    st.write(f"**Question {evaluation_state['current_index'] + 1} of {len(evaluation_state['selected_questions'])}**")
    
    # Display the therapist question
    st.subheader("Therapist Question:")
    st.write(current_question)
    
    # Display each model response with evaluation form
    model_responses = {}
    for letter, model_name in evaluation_state["model_mapping"][current_question].items():
        response = responses[current_question][model_name]
        model_responses[letter] = {
            "model_name": model_name,
            "response": response
        }
    
    # Create tabs for each response
    tabs = st.tabs([f"Patient Response {letter}" for letter in model_responses.keys()])
    
    for i, (letter, tab) in enumerate(zip(model_responses.keys(), tabs)):
        with tab:
            st.write("**Patient Response:**")
            st.write(model_responses[letter]["response"])
            
            st.write("---")
            st.write("**Evaluation:**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                active_listening = st.slider(
                    "Active Listening (1-10):", 
                    1, 10, 5, 
                    help="How well the patient demonstrates understanding of the therapist's questions",
                    key=f"al_{evaluation_state['current_index']}_{letter}"
                )
                
                empathy = st.slider(
                    "Emotional Expression (1-10):", 
                    1, 10, 5, 
                    help="How authentically the patient expresses emotions",
                    key=f"em_{evaluation_state['current_index']}_{letter}"
                )
                
                maladaptive = st.slider(
                    "Maladaptive Cognitions (1-10):", 
                    1, 10, 5, 
                    help="How accurately the patient portrays cognitive distortions typical in depression/anxiety",
                    key=f"mc_{evaluation_state['current_index']}_{letter}"
                )
                
                emotional_states = st.slider(
                    "Emotional States (1-10):", 
                    1, 10, 5, 
                    help="How realistically the patient expresses emotions associated with depression/anxiety",
                    key=f"es_{evaluation_state['current_index']}_{letter}"
                )
            
            with col2:
                conversational = st.slider(
                    "Conversational Style (1-10):", 
                    1, 10, 5, 
                    help="How well the patient mimics natural human conversation patterns",
                    key=f"cs_{evaluation_state['current_index']}_{letter}"
                )
                
                fidelity = st.slider(
                    "Overall Fidelity (1-10):", 
                    1, 10, 5, 
                    help="How convincingly the AI simulates a real patient",
                    key=f"of_{evaluation_state['current_index']}_{letter}"
                )
                
                believability = st.slider(
                    "Clinical Believability (1-10):", 
                    1, 10, 5, 
                    help="How believable the patient's presentation is from a clinical perspective",
                    key=f"cb_{evaluation_state['current_index']}_{letter}"
                )
            
            comments = st.text_area(
                "Comments (specific strengths/weaknesses):", 
                key=f"comments_{evaluation_state['current_index']}_{letter}"
            )
            
            educational_value = st.slider(
                "Educational Value for Trainees (1-10):", 
                1, 10, 5, 
                help="How valuable this simulated patient would be for training purposes",
                key=f"ev_{evaluation_state['current_index']}_{letter}"
            )
            
            # Store evaluation
            eval_key = f"{current_question}_{letter}"
            evaluation_state["evaluations"][eval_key] = {
                "question": current_question,
                "response": model_responses[letter]["response"],
                "model_letter": letter,
                "actual_model": model_responses[letter]["model_name"],
                "active_listening": active_listening,
                "emotional_expression": empathy,
                "maladaptive_cognitions": maladaptive,
                "emotional_states": emotional_states,
                "conversational_style": conversational,
                "overall_fidelity": fidelity,
                "clinical_believability": believability,
                "educational_value": educational_value,
                "comments": comments,
                "evaluator": expert_info
            }
    
    # Navigation buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Previous Question", disabled=evaluation_state["current_index"] == 0):
            evaluation_state["current_index"] -= 1
            st.rerun()
    
    with col2:
        if evaluation_state["current_index"] < len(evaluation_state["selected_questions"]) - 1:
            next_text = "Next Question"
        else:
            next_text = "Finish Evaluation"
        
        if st.button(next_text):
            evaluation_state["current_index"] += 1
            st.rerun()

def show_results(evaluation_state, expert_info):
    st.header("Evaluation Complete")
    st.write("Thank you for completing the evaluation! Here's a summary of your assessments:")
    
    # Convert evaluations to DataFrame
    eval_data = []
    for eval_key, evaluation in evaluation_state["evaluations"].items():
        eval_data.append({
            "Model": evaluation["actual_model"],
            "Active Listening": evaluation["active_listening"],
            "Emotional Expression": evaluation["emotional_expression"],
            "Maladaptive Cognitions": evaluation["maladaptive_cognitions"],
            "Emotional States": evaluation["emotional_states"],
            "Conversational Style": evaluation["conversational_style"],
            "Overall Fidelity": evaluation["overall_fidelity"],
            "Clinical Believability": evaluation["clinical_believability"],
            "Educational Value": evaluation["educational_value"]
        })
    
    eval_df = pd.DataFrame(eval_data)
    
    # Group by model and calculate average scores
    model_summary = eval_df.groupby("Model").mean().reset_index()
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    melted_df = pd.melt(model_summary, id_vars=["Model"], var_name="Metric", value_name="Score")
    
    sns.barplot(data=melted_df, x="Model", y="Score", hue="Metric", ax=ax)
    ax.set_title("Your Evaluation Scores by Model and Metric")
    ax.set_xlabel("Model")
    ax.set_ylabel("Average Score (1-10)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    st.pyplot(fig)
    
    # Display overall ranking
    st.subheader("Overall Model Ranking")
    model_ranking = model_summary.sort_values("Overall Fidelity", ascending=False)
    
    for i, (_, row) in enumerate(model_ranking.iterrows()):
        st.write(f"{i+1}. **{row['Model']}** - Overall Fidelity: {row['Overall Fidelity']:.2f}/10")
    
    # Save evaluation
    if st.button("Save and Submit Evaluation"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"evaluations/expert_{expert_info['name'].replace(' ', '_')}_{timestamp}.json"
        
        os.makedirs("evaluations", exist_ok=True)
        
        with open(filename, "w") as f:
            json.dump(evaluation_state["evaluations"], f, indent=2)
        
        st.success(f"Evaluation saved successfully to {filename}")
        st.balloons()

def main():
    setup_page()
    
    # Get expert information
    expert_info = expert_information()
    
    # Load or generate model responses
    responses, response_type = load_or_generate_responses()
    
    if not responses:
        st.warning("No responses available. Please configure models and generate responses.")
        return
    
    # Validate expert info
    if not expert_info["name"]:
        st.warning("Please enter your name in the sidebar before proceeding.")
        return
    
    # Prepare evaluation session
    evaluation_state = prepare_evaluation_session(responses)
    
    # Show either evaluation or results
    if evaluation_state["completed"]:
        show_results(evaluation_state, expert_info)
    else:
        evaluate_responses(responses, evaluation_state, expert_info)

if __name__ == "__main__":
    main()