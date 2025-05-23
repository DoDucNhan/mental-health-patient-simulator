# trainee_evaluation_app.py
import streamlit as st
import pandas as pd
import json
import random
import os
from datetime import datetime
from mental_health_patient_instructions import get_depression_instructions, get_anxiety_instructions

def setup_page():
    st.set_page_config(
        page_title="Mental Health Training Simulation",
        page_icon="🧠",
        layout="wide"
    )
    
    st.title("Mental Health Training Simulation")
    st.write("""
    Practice your therapeutic skills with AI-simulated patients. This tool helps mental health 
    trainees develop their assessment and cognitive model formulation abilities in a safe environment.
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

def generate_patient_response(model, tokenizer, therapist_input, condition_type, conversation_style, chat_history):
    """Generate patient response using the loaded model"""
    if model is None or tokenizer is None:
        return "I'm sorry, I'm having trouble responding right now."
    
    try:
        # Get appropriate instructions
        if condition_type == "depression":
            instructions = get_depression_instructions(conversation_style)
        else:
            instructions = get_anxiety_instructions(conversation_style)
        
        # Build conversation context
        conversation_context = ""
        for message in chat_history[-3:]:  # Last 3 exchanges for context
            if message["role"] == "user":
                conversation_context += f"Therapist: {message['content']}\n"
            else:
                conversation_context += f"Patient: {message['content']}\n"
        
        # Create the full prompt
        prompt = f"{instructions}\n\nConversation so far:\n{conversation_context}Therapist: {therapist_input}\nPatient:"
        
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
        
        # Clean up response (remove any unwanted prefixes)
        if response.startswith("Patient:"):
            response = response[8:].strip()
        
        return response
    except Exception as e:
        return f"I'm having difficulty expressing myself right now... [Error: {str(e)}]"

def trainee_information():
    st.sidebar.header("Trainee Information")
    
    trainee_name = st.sidebar.text_input("Your Name")
    
    trainee_level = st.sidebar.selectbox(
        "Training Level", 
        ["Medical Student", "Psychiatry Resident", "Psychology Student", 
         "Social Work Student", "Counseling Student", "Other"]
    )
    
    prior_experience = st.sidebar.selectbox(
        "Prior Experience with Real Patients",
        ["None", "1-5 patients", "6-20 patients", "More than 20 patients"]
    )
    
    return {
        "name": trainee_name,
        "level": trainee_level,
        "experience": prior_experience
    }

def model_configuration():
    st.sidebar.header("Patient Simulation Configuration")
    
    # Model selection
    model_path = st.sidebar.text_input(
        "Model Path:",
        value="/path/to/your/finetuned/model",
        help="Enter the path to your fine-tuned model"
    )
    
    # Session configuration
    session_type = st.sidebar.selectbox(
        "Patient Condition:",
        ["depression", "anxiety"]
    )
    
    conversation_style = st.sidebar.selectbox(
        "Patient Conversation Style:",
        ["plain", "reserved", "verbose", "upset", "tangent", "pleasing"],
        help="Choose how the patient communicates"
    )
    
    return {
        "model_path": model_path,
        "session_type": session_type,
        "conversation_style": conversation_style
    }

def initialize_session_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "model_loaded" not in st.session_state:
        st.session_state.model_loaded = False
        st.session_state.current_model = None
        st.session_state.current_tokenizer = None
    
    if "session_started" not in st.session_state:
        st.session_state.session_started = False

def start_session(config):
    """Initialize the therapy session"""
    if not st.session_state.model_loaded or st.session_state.current_model_path != config["model_path"]:
        with st.spinner("Loading patient simulation model..."):
            model, tokenizer = load_model(config["model_path"])
            
            if model is not None and tokenizer is not None:
                st.session_state.current_model = model
                st.session_state.current_tokenizer = tokenizer
                st.session_state.current_model_path = config["model_path"]
                st.session_state.model_loaded = True
                st.success("Model loaded successfully!")
            else:
                st.error("Failed to load model. Please check the path and try again.")
                return False
    
    # Initialize patient background
    if not st.session_state.session_started:
        # Generate patient background based on condition
        if config["session_type"] == "depression":
            background = generate_depression_background(config["conversation_style"])
        else:
            background = generate_anxiety_background(config["conversation_style"])
        
        st.session_state.patient_background = background
        st.session_state.session_config = config
        st.session_state.session_started = True
        
        # Add initial greeting
        greeting = generate_initial_greeting(config["session_type"], config["conversation_style"])
        st.session_state.chat_history = [
            {"role": "assistant", "content": greeting}
        ]
    
    return True

def generate_depression_background(style):
    backgrounds = {
        "plain": "This patient is a 32-year-old individual experiencing depression for the past 6 months. They have been feeling sad, hopeless, and have lost interest in activities they once enjoyed.",
        "reserved": "This patient is a 28-year-old individual with depression who tends to be quiet and hesitant to share personal information. They have been struggling with mood issues but find it difficult to open up.",
        "verbose": "This patient is a 35-year-old individual with depression who tends to provide very detailed responses and often goes into extensive background information when answering questions.",
        "upset": "This patient is a 30-year-old individual with depression who is feeling frustrated with their current situation and may show resistance to therapy or express anger about their circumstances.",
        "tangent": "This patient is a 29-year-old individual with depression who has difficulty staying focused on topics and often shifts between different subjects during conversation.",
        "pleasing": "This patient is a 26-year-old individual with depression who tends to be very agreeable and tries to give answers they think the therapist wants to hear."
    }
    return backgrounds.get(style, backgrounds["plain"])

def generate_anxiety_background(style):
    backgrounds = {
        "plain": "This patient is a 27-year-old individual experiencing generalized anxiety disorder for the past year. They worry excessively about various aspects of life and feel restless most days.",
        "reserved": "This patient is a 31-year-old individual with anxiety who is nervous about therapy and tends to give brief answers. They worry about being judged.",
        "verbose": "This patient is a 33-year-old individual with anxiety who tends to over-explain their worries and provides excessive detail about their anxious thoughts and situations.",
        "upset": "This patient is a 29-year-old individual with anxiety who is frustrated with their constant worry and may express irritation with themselves or their situation.",
        "tangent": "This patient is a 25-year-old individual with anxiety whose worried thoughts often jump from topic to topic, making it difficult to maintain focus on one concern.",
        "pleasing": "This patient is a 24-year-old individual with anxiety who worries about disappointing the therapist and tries to be the 'perfect' patient."
    }
    return backgrounds.get(style, backgrounds["plain"])

def generate_initial_greeting(session_type, style):
    if session_type == "depression":
        greetings = {
            "plain": "Hi... I'm here because I've been feeling really down lately. I'm not sure where to start.",
            "reserved": "Hello. Um... I'm not really sure what I'm supposed to say.",
            "verbose": "Hi there. Well, I've been dealing with a lot lately, and my doctor suggested I talk to someone because I've been feeling really depressed for months now, and it's affecting my work and relationships...",
            "upset": "I'm here because everyone keeps telling me I need help. I don't know if this is going to work.",
            "tangent": "Hi... so I'm here about feeling depressed, but actually, it reminds me of when I was younger and my grandmother used to...",
            "pleasing": "Hello! Thank you so much for seeing me. I hope I'm not taking up too much of your time. I'll try to be a good patient."
        }
    else:  # anxiety
        greetings = {
            "plain": "Hi... I've been having a lot of anxiety lately and it's getting hard to manage.",
            "reserved": "Hello. I... I get really nervous talking about this stuff.",
            "verbose": "Hi! So I've been experiencing what I think is anxiety - my heart races, I can't sleep, I worry about everything constantly, and it started about six months ago when...",
            "upset": "I'm here because my anxiety is ruining my life and nothing I try seems to work.",
            "tangent": "Hi, so I'm anxious about this session, which reminds me that I'm anxious about everything lately, like this morning I was worried about being late, which made me think about...",
            "pleasing": "Hello! I'm so sorry if I seem nervous. I don't want to waste your time, but I've been having some anxiety issues."
        }
    
    return greetings.get(style, greetings["plain"])

def display_chat_history():
    """Display the conversation history"""
    st.subheader("Therapy Session")
    
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.write(f"**Therapist:** {message['content']}")
        else:
            st.write(f"**Patient:** {message['content']}")
        st.write("---")

def therapy_session_interface():
    """Main therapy session interface"""
    # Display patient background
    if "patient_background" in st.session_state:
        with st.expander("Patient Background (for reference)"):
            st.write(st.session_state.patient_background)
    
    # Display conversation
    display_chat_history()
    
    # Input for therapist message
    therapist_input = st.text_area(
        "Your response as therapist:",
        key="therapist_message",
        help="Enter your therapeutic response or question"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Send Message"):
            if therapist_input.strip():
                # Add therapist message to history
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": therapist_input
                })
                
                # Generate patient response
                with st.spinner("Patient is thinking..."):
                    config = st.session_state.session_config
                    patient_response = generate_patient_response(
                        st.session_state.current_model,
                        st.session_state.current_tokenizer,
                        therapist_input,
                        config["session_type"],
                        config["conversation_style"],
                        st.session_state.chat_history
                    )
                
                # Add patient response to history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": patient_response
               })
               
                # Clear the input and rerun
                st.rerun()

    with col2:
        if st.button("Reset Session"):
            st.session_state.chat_history = []
            st.session_state.session_started = False
            st.session_state.cognitive_model_formulated = False
            st.rerun()

def cognitive_model_formulation():
   """Interface for cognitive model formulation"""
   if len(st.session_state.chat_history) >= 6:  # After 3 exchanges
       st.write("---")
       st.header("Cognitive Model Formulation")
       st.write("Based on your conversation with the simulated patient, please formulate a cognitive model:")
       
       col1, col2 = st.columns(2)
       
       with col1:
           # Core beliefs
           core_beliefs = st.text_area(
               "Core Beliefs:",
               help="Identify the patient's fundamental beliefs about self, others, and the world",
               key="core_beliefs"
           )
           
           # Intermediate beliefs
           intermediate_beliefs = st.text_area(
               "Intermediate Beliefs:",
               help="Identify the patient's rules, attitudes, and assumptions",
               key="intermediate_beliefs"
           )
           
           # Automatic thoughts
           automatic_thoughts = st.text_area(
               "Automatic Thoughts:",
               help="Identify the patient's spontaneous thoughts in specific situations",
               key="automatic_thoughts"
           )
       
       with col2:
           # Emotional responses
           emotional_responses = st.text_area(
               "Emotional Responses:",
               help="Identify the patient's emotional reactions",
               key="emotional_responses"
           )
           
           # Behavioral patterns
           behavioral_patterns = st.text_area(
               "Behavioral Patterns:",
               help="Identify the patient's behavioral responses and coping strategies",
               key="behavioral_patterns"
           )
           
           # Triggers/situations
           triggers = st.text_area(
               "Triggers/Situations:",
               help="Identify situations that activate the patient's cognitive patterns",
               key="triggers"
           )
       
       # Confidence assessment
       confidence = st.slider(
           "Rate your confidence in this formulation (1-10):",
           1, 10, 5,
           key="formulation_confidence"
       )

def post_session_evaluation():
   """Post-session evaluation interface"""
   if len(st.session_state.chat_history) >= 6:
       st.write("---")
       st.header("Training Session Evaluation")
       
       col1, col2 = st.columns(2)
       
       with col1:
           realism = st.slider(
               "How realistic was this patient simulation? (1-10)",
               1, 10, 5,
               key="realism_rating"
           )
           
           helpfulness = st.slider(
               "How helpful was this for your training? (1-10)",
               1, 10, 5,
               key="helpfulness_rating"
           )
           
           difficulty = st.slider(
               "How challenging was this patient? (1-10)",
               1, 10, 5,
               key="difficulty_rating"
           )
       
       with col2:
           engagement = st.slider(
               "How engaging was the interaction? (1-10)",
               1, 10, 5,
               key="engagement_rating"
           )
           
           learning_value = st.slider(
               "How much did you learn from this session? (1-10)",
               1, 10, 5,
               key="learning_rating"
           )
           
           would_recommend = st.slider(
               "Would you recommend this to other trainees? (1-10)",
               1, 10, 5,
               key="recommend_rating"
           )
       
       feedback = st.text_area(
           "Additional feedback about the simulation:",
           key="additional_feedback"
       )
       
       # Areas for improvement
       st.subheader("What could be improved?")
       improvements = st.multiselect(
           "Select areas that could be improved:",
           [
               "More realistic responses",
               "Better emotional expression",
               "More varied conversation styles",
               "Clearer symptom presentation",
               "Better context awareness",
               "More challenging scenarios",
               "Faster response time",
               "Other"
           ],
           key="improvement_areas"
       )
       
       if "Other" in improvements:
           other_improvement = st.text_input("Please specify:", key="other_improvement")

def save_session_data(trainee_info):
   """Save all session data"""
   if st.button("Submit Evaluation and Save Session"):
       # Compile all session data
       session_data = {
           "trainee_info": trainee_info,
           "session_config": st.session_state.get("session_config", {}),
           "patient_background": st.session_state.get("patient_background", ""),
           "chat_history": st.session_state.chat_history,
           "cognitive_model": {
               "core_beliefs": st.session_state.get("core_beliefs", ""),
               "intermediate_beliefs": st.session_state.get("intermediate_beliefs", ""),
               "automatic_thoughts": st.session_state.get("automatic_thoughts", ""),
               "emotional_responses": st.session_state.get("emotional_responses", ""),
               "behavioral_patterns": st.session_state.get("behavioral_patterns", ""),
               "triggers": st.session_state.get("triggers", ""),
               "confidence": st.session_state.get("formulation_confidence", 5)
           },
           "evaluation": {
               "realism": st.session_state.get("realism_rating", 5),
               "helpfulness": st.session_state.get("helpfulness_rating", 5),
               "difficulty": st.session_state.get("difficulty_rating", 5),
               "engagement": st.session_state.get("engagement_rating", 5),
               "learning_value": st.session_state.get("learning_rating", 5),
               "would_recommend": st.session_state.get("recommend_rating", 5),
               "feedback": st.session_state.get("additional_feedback", ""),
               "improvement_areas": st.session_state.get("improvement_areas", []),
               "other_improvement": st.session_state.get("other_improvement", "")
           },
           "timestamp": datetime.now().isoformat()
       }
       
       # Save to file
       os.makedirs("trainee_sessions", exist_ok=True)
       timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
       filename = f"trainee_sessions/session_{trainee_info['name'].replace(' ', '_')}_{timestamp}.json"
       
       with open(filename, "w") as f:
           json.dump(session_data, f, indent=2)
       
       st.success(f"Session saved successfully to {filename}")
       st.balloons()
       
       # Show summary
       show_session_summary(session_data)

def show_session_summary(session_data):
   """Display session summary"""
   st.header("Session Summary")
   
   col1, col2 = st.columns(2)
   
   with col1:
       st.subheader("Session Statistics")
       st.write(f"**Number of exchanges:** {len(session_data['chat_history']) // 2}")
       st.write(f"**Patient condition:** {session_data['session_config']['session_type']}")
       st.write(f"**Conversation style:** {session_data['session_config']['conversation_style']}")
       st.write(f"**Formulation confidence:** {session_data['cognitive_model']['confidence']}/10")
   
   with col2:
       st.subheader("Evaluation Scores")
       st.write(f"**Realism:** {session_data['evaluation']['realism']}/10")
       st.write(f"**Helpfulness:** {session_data['evaluation']['helpfulness']}/10")
       st.write(f"**Learning value:** {session_data['evaluation']['learning_value']}/10")
       st.write(f"**Would recommend:** {session_data['evaluation']['would_recommend']}/10")

def main():
   setup_page()
   
   # Initialize session state
   initialize_session_state()
   
   # Get trainee information
   trainee_info = trainee_information()
   
   # Get model configuration
   config = model_configuration()
   
   # Validate inputs
   if not trainee_info["name"]:
       st.warning("Please enter your name in the sidebar before starting the session.")
       return
   
   if not config["model_path"] or config["model_path"] == "/path/to/your/finetuned/model":
       st.warning("Please enter a valid model path in the sidebar.")
       return
   
   # Start session button
   if not st.session_state.session_started:
       if st.button("Start Therapy Session", type="primary"):
           if start_session(config):
               st.success("Session started! You can now begin the conversation.")
               st.rerun()
   else:
       # Display current session info
       st.info(f"**Current session:** {config['session_type'].title()} patient with {config['conversation_style']} conversation style")
       
       # Main therapy interface
       therapy_session_interface()
       
       # Cognitive model formulation
       cognitive_model_formulation()
       
       # Post-session evaluation
       post_session_evaluation()
       
       # Save session data
       save_session_data(trainee_info)

if __name__ == "__main__":
   main()