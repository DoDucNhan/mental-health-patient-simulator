import random
import json

def create_test_questions():
   """Create a set of test questions for evaluating mental health patient simulations"""
   
   # Categories of therapist questions
   categories = {
       "symptom_exploration": [
           "Can you tell me more about how your mood has been over the past week?",
           "How would you describe your energy levels lately?",
           "Have you noticed any changes in your sleep patterns?",
           "Are there specific situations that trigger your anxiety?",
           "Can you describe how these thoughts affect your daily life?",
           "How often do you experience these feelings of hopelessness?",
           "Can you tell me about any physical symptoms you experience with your anxiety?",
           "Have you noticed any changes in your appetite or eating habits?",
           "How would you describe your concentration and focus lately?",
           "Do you find yourself avoiding any specific situations or activities?"
       ],
       "thought_patterns": [
           "What thoughts go through your mind when you're feeling anxious?",
           "Do you notice any patterns in how you talk to yourself?",
           "What are some of the negative thoughts you have about yourself?",
           "When you face a challenging situation, what's your first thought?",
           "Do you ever feel like your thoughts are racing and hard to control?",
           "What do you think will happen if you fail at something important to you?",
           "How do you typically respond to criticism from others?",
           "Do you find yourself expecting the worst in situations?",
           "What beliefs do you hold about yourself that might not be entirely accurate?",
           "How do you interpret it when someone doesn't respond to your message?"
       ],
       "emotional_responses": [
           "How do you usually cope with feelings of sadness?",
           "What emotions do you find most difficult to manage?",
           "Can you describe a recent situation where you felt overwhelmed?",
           "How do you typically respond when you feel anxious in social situations?",
           "When was the last time you felt truly happy or content?",
           "Do you ever experience feelings that seem out of proportion to the situation?",
           "How do you usually express your anger or frustration?",
           "Do you ever feel numb or emotionally disconnected?",
           "What brings you joy or comfort during difficult times?",
           "How would you describe your ability to recognize your own emotions?"
       ],
       "behavioral_patterns": [
           "What strategies have you developed to cope with your depression?",
           "Have you noticed any changes in your social interactions lately?",
           "How has your motivation to engage in activities you usually enjoy been affected?",
           "What do you typically do when you start feeling anxious?",
           "Have friends or family commented on changes in your behavior?",
           "How do your symptoms affect your ability to work or study?",
           "Have you developed any routines or rituals to manage your anxiety?",
           "Do you find yourself withdrawing from people when you're feeling down?",
           "How has your depression affected your self-care habits?",
           "What activities have you stopped doing because of how you've been feeling?"
       ],
       "interpersonal_relationships": [
           "How have your relationships been affected by your depression?",
           "Do you find it difficult to express your needs to others?",
           "How do you respond when you feel misunderstood by someone close to you?",
           "Has your anxiety impacted how you interact with others?",
           "Do you find it hard to trust people or open up about your feelings?",
           "How do you handle conflict in your relationships?",
           "Do you worry about being a burden to others with your problems?",
           "How do you feel about asking for help when you need it?",
           "Have you noticed any patterns in your relationships that concern you?",
           "How comfortable are you with setting boundaries with others?"
       ],
       "challenging_questions": [
           "What's stopping you from making the changes we've discussed?",
           "How do you think your current coping strategies are working for you?",
           "What would happen if you allowed yourself to feel these emotions fully?",
           "What's your biggest fear about getting better?",
           "How might things be different if you didn't believe that thought?",
           "What does recovery mean to you?",
           "What would you say to a friend who was thinking the way you are now?",
           "What do you think you need most right now to help you move forward?",
           "How might you be contributing to some of the problems we've discussed?",
           "What beliefs about yourself would you need to change to feel better?"
       ],
       "supportive_questions": [
           "What has helped you get through difficult times in the past?",
           "What strengths do you have that could help you now?",
           "Who in your life provides you with support?",
           "What self-care activities help you feel better?",
           "What small step could you take this week toward feeling better?",
           "What's one thing you're proud of accomplishing despite your depression?",
           "What gives you hope even when things are difficult?",
           "What has improved, even slightly, since we last spoke?",
           "What would a compassionate response to your situation look like?",
           "What resources or support do you feel might be helpful right now?"
       ]
   }
   
   # Generate a balanced set of questions
   all_questions = []
   for category, questions in categories.items():
       all_questions.extend(questions)
   
   # Add some specific depression and anxiety scenario questions
   depression_scenarios = [
       "I noticed you mentioned feeling worthless. Could you tell me more about these thoughts?",
       "You said you've been having trouble getting out of bed in the morning. What's that experience like for you?",
       "How has your depression affected your view of the future?",
       "You mentioned feeling like a burden to others. Could you elaborate on that?",
       "What aspects of your depression do you find most difficult to manage?",
       "How do you see your depression now compared to when it first started?",
       "In what ways has depression changed how you see yourself?",
       "What do you think contributes most to maintaining your depression?",
       "How does your depression affect your ability to find meaning or purpose?",
       "What's your experience with the ups and downs of depression?"
   ]
   
   anxiety_scenarios = [
       "You mentioned having panic attacks. Could you walk me through what happens during a typical episode?",
       "How does your anxiety affect your decision-making process?",
       "What safety behaviors have you developed to manage your anxiety?",
       "You talked about worry consuming your thoughts. What kinds of things do you worry about most?",
       "How does your body respond when you're in an anxiety-provoking situation?",
       "What effect has living with chronic anxiety had on your overall outlook?",
       "When you notice anxiety building, what typically happens next?",
       "How has your anxiety affected your willingness to try new things?",
       "What things do you avoid because of your anxiety?",
       "How predictable are your anxiety symptoms?"
   ]
   
   all_questions.extend(depression_scenarios)
   all_questions.extend(anxiety_scenarios)
   
   # Add some follow-up questions that would test conversation coherence
   follow_up_scenarios = [
       "Earlier you mentioned struggling with sleep. How does that affect your next day?",
       "You talked about feeling judged by others. Can you give a recent example?",
       "Going back to what you said about your family situation, how do you think that relates to your current feelings?",
       "You mentioned having recurring thoughts about not being good enough. When did these thoughts first start?",
       "I'm curious about the coping mechanism you described. How effective has that been for you?",
       "Let's explore more about that childhood experience you mentioned. How do you think it shaped your current beliefs?",
       "You briefly touched on your relationship with your parents. Could you tell me more about that?",
       "When you described that feeling of emptiness, what else accompanies that sensation?",
       "You mentioned feeling overwhelmed at work. What specifically about your work environment triggers these feelings?",
       "Earlier you said something about having high standards for yourself. How does that manifest in your daily life?"
   ]
   
   all_questions.extend(follow_up_scenarios)
   
   # Shuffle and select 200 questions
   random.shuffle(all_questions)
   selected_questions = all_questions[:200] if len(all_questions) > 200 else all_questions
   
   # Save to file
   with open("test_questions.json", "w") as f:
       json.dump(selected_questions, f, indent=2)
   
   print(f"Created {len(selected_questions)} test questions.")
   return selected_questions

if __name__ == "__main__":
   test_questions = create_test_questions()