import streamlit as st
import pandas as pd
import json
import random
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

def load_responses():
    # Get all response files
    responses_dir = "responses"
    response_files = [f for f in os.listdir(responses_dir) if f.endswith(".json")]
    
    if not response_files:
        st.error("No response files found. Please generate responses first.")
        st.stop()
    
    # Select the most recent file by default
    response_files.sort(reverse=True)
    selected_file = st.selectbox(
        "Select response file to evaluate:", 
        response_files,
        index=0
    )
    
    # Load the selected file
    with open(os.path.join(responses_dir, selected_file), "r") as f:
        responses = json.load(f)
    
    return responses, selected_file

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
            st.experimental_rerun()
    
    with col2:
        if evaluation_state["current_index"] < len(evaluation_state["selected_questions"]) - 1:
            next_text = "Next Question"
        else:
            next_text = "Finish Evaluation"
        
        if st.button(next_text):
            evaluation_state["current_index"] += 1
            st.experimental_rerun()

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
    plt.figure(figsize=(12, 8))
    melted_df = pd.melt(model_summary, id_vars=["Model"], var_name="Metric", value_name="Score")
    
    plt.figure(figsize=(14, 8))
    chart = sns.barplot(x="Model", y="Score", hue="Metric", data=melted_df)
    plt.title("Your Evaluation Scores by Model and Metric")
    plt.xlabel("Model")
    plt.ylabel("Average Score (1-10)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Convert plot to image for Streamlit
    fig = chart.get_figure()
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
    
    # Load model responses
    responses, selected_file = load_responses()
    
    # Prepare evaluation session
    evaluation_state = prepare_evaluation_session(responses)
    
    # Validate expert info
    if not expert_info["name"]:
        st.warning("Please enter your name in the sidebar before proceeding.")
        return
    
    # Show either evaluation or results
    if evaluation_state["completed"]:
        show_results(evaluation_state, expert_info)
    else:
        evaluate_responses(responses, evaluation_state, expert_info)

if __name__ == "__main__":
    main()