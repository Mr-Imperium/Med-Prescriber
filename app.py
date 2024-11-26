import streamlit as st

# Set page config as the first command
st.set_page_config(page_title="Disease Prediction App", layout="wide")

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle

# Load the model and data
@st.cache_resource
def load_model_and_data():
    # Load the trained model
    with open('svc.pkl', 'rb') as f:
        svc = pickle.load(f)
    
    # Load datasets
    description = pd.read_csv("data/description.csv")
    precautions = pd.read_csv("data/precautions_df.csv")
    workout = pd.read_csv("data/workout_df.csv")
    medications = pd.read_csv("data/medications.csv")
    diets = pd.read_csv("data/diets.csv")
    
    # Load training data to get symptoms
    dataset = pd.read_csv('data/Training.csv')
    symptoms = list(dataset.drop('prognosis', axis=1).columns)
    
    return svc, symptoms, description, precautions, workout, medications, diets

# Prediction function
def get_predicted_value(svc, symptoms_dict, patient_symptoms):
    # Create input vector based on all possible symptoms
    input_vector = np.zeros(len(symptoms_dict))
    
    # Mark input symptoms as present
    for item in patient_symptoms:
        input_vector[symptoms_dict[item]] = 1
    
    # Predict disease
    predicted_disease = svc.predict([input_vector])[0]
    return predicted_disease

# Helper function to get disease details
def get_disease_details(disease, description, precautions, workout, medications, diets):
    # Description
    desc = description[description['Disease'] == disease]['Description'].values
    desc = desc[0] if len(desc) > 0 else "No description available."
    
    # Precautions
    pre_df = precautions[precautions['Disease'] == disease]
    pre = []
    if not pre_df.empty:
        pre = [pre_df['Precaution_1'].values[0], 
               pre_df['Precaution_2'].values[0], 
               pre_df['Precaution_3'].values[0], 
               pre_df['Precaution_4'].values[0]]
        pre = [p for p in pre if p and str(p).strip()]
    
    # Medications
    med_df = medications[medications['Disease'] == disease]
    med = med_df['Medication'].tolist() if not med_df.empty else []
    
    # Diets
    die_df = diets[diets['Disease'] == disease]
    die = die_df['Diet'].tolist() if not die_df.empty else []
    
    # Workout
    wrkout_df = workout[workout['disease'] == disease]
    wrkout = wrkout_df['workout'].tolist() if not wrkout_df.empty else []
    
    return desc, pre, med, die, wrkout

# Main Streamlit App
def main():
    # Load model and data
    svc, all_symptoms, description, precautions, workout, medications, diets = load_model_and_data()
    
    # Create symptoms dictionary
    symptoms_dict = {symptom: index for index, symptom in enumerate(all_symptoms)}
    
    # Create tabs
    tab1, tab2 = st.tabs(["Symptom Checker", "Disease Information"])
    
    # Symptom Checker Tab
    with tab1:
        st.title("Disease Prediction from Symptoms")
        
        # Multiselect for symptoms
        selected_symptoms = st.multiselect(
            "Select your symptoms", 
            all_symptoms,
            placeholder="Choose symptoms"
        )
        
        # Prediction button
        if st.button("Predict Disease"):
            if selected_symptoms:
                # Get predicted disease
                predicted_disease = get_predicted_value(svc, symptoms_dict, selected_symptoms)
                
                # Get disease details
                desc, pre, med, die, wrkout = get_disease_details(
                    predicted_disease, 
                    description, 
                    precautions, 
                    workout, 
                    medications, 
                    diets
                )
                
                # Display results
                st.subheader(f"Predicted Disease: {predicted_disease}")
                
                # Description
                st.markdown("### Description")
                st.write(desc)
                
                # Precautions
                st.markdown("### Precautions")
                if pre:
                    for p in pre:
                        st.write(f"- {p}")
                else:
                    st.write("No precautions available.")
                
                # Medications
                st.markdown("### Medications")
                if med:
                    for m in med:
                        st.write(f"- {m}")
                else:
                    st.write("No medications information available.")
                
                # Workout
                st.markdown("### Recommended Workouts")
                if wrkout:
                    for w in wrkout:
                        st.write(f"- {w}")
                else:
                    st.write("No workout information available.")
                
                # Diets
                st.markdown("### Recommended Diets")
                if die:
                    for d in die:
                        st.write(f"- {d}")
                else:
                    st.write("No diet information available.")
            else:
                st.warning("Please select at least one symptom.")
    
    # Disease Information Tab
    with tab2:
        st.title("Disease Information Database")
        
        # Get unique diseases
        unique_diseases = description['Disease'].unique()
        
        # Create a dataframe for display
        disease_df = pd.DataFrame({
            'Disease': unique_diseases,
            'Description': [description[description['Disease'] == disease]['Description'].values[0] 
                            for disease in unique_diseases]
        })
        
        # Display table with disease info
        st.dataframe(
            disease_df, 
            column_config={
                "Disease": st.column_config.TextColumn("Disease Name"),
                "Description": st.column_config.TextColumn("Short Description", width="large")
            },
            hide_index=True,
            use_container_width=True
        )

# Run the app
if __name__ == "__main__":
    main()
