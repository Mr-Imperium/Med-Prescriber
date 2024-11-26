import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle

# Set page config as the first command
st.set_page_config(page_title="Disease Prediction App", layout="wide")

# Load the model and data
@st.cache_resource
def load_model_and_data():
    # Load the trained model
    with open('svc.pkl', 'rb') as f:
        svc = pickle.load(f)
    
    # Load the label encoder
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    
    # Load datasets
    description = pd.read_csv("data/description.csv")
    precautions = pd.read_csv("data/precautions_df.csv")
    workout = pd.read_csv("data/workout_df.csv")
    medications = pd.read_csv("data/medications.csv")
    diets = pd.read_csv("data/diets.csv")
    
    # Load training data to get symptoms
    dataset = pd.read_csv('data/Training.csv')
    symptoms = list(dataset.drop('prognosis', axis=1).columns)
    
    return svc, le, symptoms, description, precautions, workout, medications, diets

# Prediction function
def get_predicted_value(svc, le, symptoms_dict, patient_symptoms):
    # Create input vector based on all possible symptoms
    input_vector = np.zeros(len(symptoms_dict))
    
    # Mark input symptoms as present
    for item in patient_symptoms:
        input_vector[symptoms_dict[item]] = 1
    
    # Predict disease index
    predicted_index = svc.predict([input_vector])[0]
    
    # Convert index to disease name
    predicted_disease = le.inverse_transform([predicted_index])[0]
    
    return predicted_disease

# Helper function to get disease details
def get_disease_details(disease, description, precautions, workout, medications, diets):
    # Description
    desc = description[description['Disease'].str.lower().str.strip() == str(disease).lower().strip()]['Description'].values
    desc = desc[0] if len(desc) > 0 else "No description available."
    
    # Precautions
    pre_df = precautions[precautions['Disease'].str.lower().str.strip() == str(disease).lower().strip()]
    pre = []
    if not pre_df.empty:
        pre = [pre_df['Precaution_1'].values[0], 
               pre_df['Precaution_2'].values[0], 
               pre_df['Precaution_3'].values[0], 
               pre_df['Precaution_4'].values[0]]
        pre = [p for p in pre if p and str(p).strip()]
    
    # Medications
    med_df = medications[medications['Disease'].str.lower().str.strip() == str(disease).lower().strip()]
    med = med_df['Medication'].tolist() if not med_df.empty else []
    
    # Diets
    die_df = diets[diets['Disease'].str.lower().str.strip() == str(disease).lower().strip()]
    die = die_df['Diet'].tolist() if not die_df.empty else []
    
    # Workout
    wrkout_df = workout[workout['disease'].str.lower().str.strip() == str(disease).lower().strip()]
    wrkout = wrkout_df['workout'].tolist() if not wrkout_df.empty else []
    
    return desc, pre, med, die, wrkout

def create_disease_info_tab(description, precautions, workout, medications, diets):
    st.title("Disease Information Database")
    
    # Custom CSS for card-like appearance and modal
    st.markdown("""
    <style>
    .disease-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
        gap: 15px;
    }
    .disease-card {
        border: 1px solid #e6e6e6;
        border-radius: 10px;
        padding: 15px;
        background-color: rgba(255, 255, 255, 0.2);
        color: white;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        cursor: pointer;
        text-align: center;
    }
    .disease-card:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .modal {
        display: none;
        position: fixed;
        z-index: 1000;
        left: 0;
        top: 0;
        width: 100%;
        height: 100%;
        overflow: auto;
        background-color: rgba(0,0,0,0.4);
        justify-content: center;
        align-items: center;
    }
    .modal-content {
        background-color: #fefefe;
        padding: 20px;
        border-radius: 10px;
        width: 80%;
        max-width: 600px;
        max-height: 80%;
        overflow-y: auto;
        position: relative;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .close {
        color: #aaa;
        float: right;
        font-size: 28px;
        font-weight: bold;
        cursor: pointer;
    }
    .close:hover {
        color: black;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Get unique diseases with descriptions, sorted alphabetically
    disease_info = description[['Disease', 'Description']].drop_duplicates().sort_values('Disease')
    
    # Create a grid of disease cards
    st.markdown(f"<div class='disease-grid'>", unsafe_allow_html=True)
    
    # Custom JavaScript for modal functionality
    st.markdown("""
    <script>
    function openModal(diseaseId) {
        document.getElementById(diseaseId).style.display = "flex";
    }
    function closeModal(diseaseId) {
        document.getElementById(diseaseId).style.display = "none";
    }
    </script>
    """, unsafe_allow_html=True)
    
    # Populate grid with disease cards
    for _, disease in disease_info.iterrows():
        # Unique modal ID
        modal_id = f"modal_{disease['Disease'].replace(' ', '_')}"
        
        # Generate disease details
        desc, pre, med, die, wrkout = get_disease_details(
            disease['Disease'], 
            description, 
            precautions, 
            workout, 
            medications, 
            diets
        )
        
        # Card HTML
        st.markdown(f"""
        <div class="disease-card" onclick="openModal('{modal_id}')">
            {disease['Disease']}
        </div>

        <!-- Modal -->
        <div id='{modal_id}' class="modal" style="display: none;">
            <div class="modal-content">
                <span class="close" onclick="closeModal('{modal_id}')">&times;</span>
                <h2>{disease['Disease']}</h2>
                
                <h3>Description</h3>
                <p>{desc}</p>
                
                <h3>Precautions</h3>
                {"".join(f"<p>- {p}</p>" for p in pre) if pre else "<p>No precautions available.</p>"}
                
                <h3>Medications</h3>
                {"".join(f"<p>- {m}</p>" for m in med) if med else "<p>No medications information available.</p>"}
                
                <h3>Recommended Non-Pharmacological Measures</h3>
                {"".join(f"<p>- {w}</p>" for w in wrkout) if wrkout else "<p>No workout information available.</p>"}
                
                <h3>Recommended Diets</h3>
                {"".join(f"<p>- {d}</p>" for d in die) if die else "<p>No diet information available.</p>"}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
def main():
    # Load model and data
    svc, le, all_symptoms, description, precautions, workout, medications, diets = load_model_and_data()
    
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
                predicted_disease = get_predicted_value(svc, le, symptoms_dict, selected_symptoms)
                
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
                st.markdown("### Recommended Non-Pharmacological Measures")
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
    
    # Disease Information tab
    with tab2:
        create_disease_info_tab(description, precautions, workout, medications, diets)

# Run the app
if __name__ == "__main__":
    main()
