import streamlit as st
import pandas as pd
import json
import random
from datetime import datetime

def create_trainee_evaluation_app():
    st.title("Mental Health Training Simulation")
    
    # Trainee information
    st.header("Trainee Information")
    trainee_name = st.text_input("Your Name")
    trainee_level = st.selectbox(
        "Training Level", 
        ["Medical Student", "Psychiatry Resident", "Psychology Student", "Social Work Student", "Other"]
    )
    prior_experience = st.selectbox(
        "Prior Experience with Real Patients",
        ["None", "1-5 patients", "6-20 patients", "More than 20 patients"]
    )
    
    # Model selection
    st.header("Simulated Patient Selection")
    model_options = [
        "OpenHermes-2.5-Mistral", 
        "Phi-3-mini", 
        "Gemma-7B-Instruct", 
        "Llama-3-8B-Instruct"
    ]
    selected_model = st.selectbox("Select Patient Simulation Model", model_options)
    
    # Session type
    session_type = st.selectbox(
        "Select Session Type",
        ["Depression", "Anxiety", "Mixed Depression and Anxiety"]
    )
    
    # Conversation style
    conversation_style = st.selectbox(
        "Select Patient Conversation Style",
        ["Plain", "Reserved", "Verbose", "Upset", "Tangent", "Pleasing"]
    )
    
    st.write("---")
    
    # Therapy session
    st.header("Therapy Session")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    for message in st.session_state.chat_history:
        role = message["role"]
        content = message["content"]
        
        if role == "user":
            st.write(f"**Therapist:** {content}")
        else:
            st.write(f"**Patient:** {content}")
    
    # Input for new message
    therapist_input = st.text_area("Your message as therapist:", key="therapist_message")
    
    if st.button("Send"):
        if therapist_input:
            # Add therapist message to chat history
            st.session_state.chat_history.append({
                "role": "user",
                "content": therapist_input
            })
            
            # Generate patient response (this would normally call your model)
            # For demo, we'll simulate a response
            patient_response = simulate_patient_response(
                therapist_input, 
                selected_model, 
                session_type, 
                conversation_style
            )
            
            # Add patient response to chat history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": patient_response
            })
            
            # Rerun to update the display
            st.experimental_rerun()
    
    # Reset button
    if st.button("Reset Conversation"):
        st.session_state.chat_history = []
        st.experimental_rerun()
    
    # After conversation, cognitive model formulation
    if len(st.session_state.chat_history) >= 6:  # After 3 exchanges
        st.write("---")
        st.header("Cognitive Model Formulation")
        st.write("Based on your conversation with the simulated patient, please formulate a cognitive model:")
        
        # Core beliefs
        core_beliefs = st.text_area("Core Beliefs:", 
                                    help="Identify the patient's fundamental beliefs about self, others, and the world")
        
        # Intermediate beliefs
        intermediate_beliefs = st.text_area("Intermediate Beliefs:", 
                                           help="Identify the patient's rules, attitudes, and assumptions")
        
        # Automatic thoughts
        automatic_thoughts = st.text_area("Automatic Thoughts:", 
                                         help="Identify the patient's spontaneous thoughts in specific situations")
        
        # Emotional responses
        emotional_responses = st.text_area("Emotional Responses:", 
                                          help="Identify the patient's emotional reactions")
        
        # Behavioral patterns
        behavioral_patterns = st.text_area("Behavioral Patterns:", 
                                          help="Identify the patient's behavioral responses and coping strategies")
        
        # Confidence assessment
        confidence = st.slider("Rate your confidence in this formulation (1-10):", 1, 10, 5)
        
        # Post-session evaluation
        st.write("---")
        st.header("Training Session Evaluation")
        
        realism = st.slider("How realistic was this patient simulation? (1-10)", 1, 10, 5)
        helpfulness = st.slider("How helpful was this for your training? (1-10)", 1, 10, 5)
        
        feedback = st.text_area("Additional feedback about the simulation:")
        
        if st.button("Submit Evaluation"):
            # Save the evaluation
            evaluation = {
                "trainee_name": trainee_name,
                "trainee_level": trainee_level,
                "prior_experience": prior_experience,
                "selected_model": selected_model,
                "session_type": session_type,
                "conversation_style": conversation_style,
                "chat_history": st.session_state.chat_history,
                "cognitive_model": {
                    "core_beliefs": core_beliefs,
                    "intermediate_beliefs": intermediate_beliefs,
                    "automatic_thoughts": automatic_thoughts,
                    "emotional_responses": emotional_responses,
                    "behavioral_patterns": behavioral_patterns
                },
                "confidence": confidence,
                "realism": realism,
                "helpfulness": helpfulness,
                "feedback": feedback,
                "timestamp": datetime.now().isoformat()
            }
            
            # Save to file
            filename = f"trainee_evaluation_{trainee_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, "w") as f:
                json.dump(evaluation, f, indent=2)
            
            st.success(f"Evaluation submitted successfully!")
            st.balloons()

def simulate_patient_response(therapist_input, model_name, session_type, conversation_style):
    """
    This function would call your actual model to generate a response.
    For demonstration purposes, we're using a placeholder.
    """
    # In a real implementation, this would use your fine-tuned model
    return f"This is a simulated response from {model_name} acting as a patient with {session_type}, using {conversation_style} conversation style."

if __name__ == "__main__":
    create_trainee_evaluation_app()