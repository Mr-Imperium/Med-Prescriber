import streamlit as st
import pandas as pd

def create_disease_info_tab(description, precautions, workout, medications, diets):
    st.title("Disease Information Database")
    
    # Custom CSS for card-like appearance
    st.markdown("""
    <style>
    .disease-card {
        border: 1px solid #e6e6e6;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        background-color: #f9f9f9;
        transition: transform 0.3s ease;
    }
    .disease-card:hover {
        transform: scale(1.03);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .disease-card h3 {
        color: #2c3e50;
        margin-bottom: 10px;
    }
    .disease-card p {
        color: #34495e;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Get unique diseases with descriptions
    disease_info = description[['Disease', 'Description']].drop_duplicates()
    
    # Create a grid of disease cards
    cols = st.columns(3)  # 3 columns of cards
    
    for idx, (_, row) in enumerate(disease_info.iterrows()):
        # Cycle through columns
        col = cols[idx % 3]
        
        with col:
            # Create a container for each disease
            with st.container():
                st.markdown(f"""
                <div class="disease-card" onclick="showDetails('{row['Disease']}')">
                    <h3>{row['Disease']}</h3>
                    <p>{row['Description'][:150]}...</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Create an expander for detailed information
                with st.expander(f"Details for {row['Disease']}"):
                    # Get disease details
                    desc, pre, med, die, wrkout = get_disease_details(
                        row['Disease'], 
                        description, 
                        precautions, 
                        workout, 
                        medications, 
                        diets
                    )
                    
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

# Modify the main function to use this new approach
def main():
    # Load model and data
    svc, all_symptoms, description, precautions, workout, medications, diets = load_model_and_data()
    
    # Create symptoms dictionary
    symptoms_dict = {symptom: index for index, symptom in enumerate(all_symptoms)}
    
    # Set up the Streamlit app
    st.set_page_config(page_title="Disease Prediction App", layout="wide")
    
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
    
    # Updated Disease Information tab
    with tab2:
        create_disease_info_tab(description, precautions, workout, medications, diets)
    
    
    
    # Disease Information Tab
    with tab2:
        st.title("Disease Information Database")
        
        # Get unique diseases
        #unique_diseases = description['Disease'].unique()
        
        # Create a dataframe for display
        #disease_df = pd.DataFrame({
            #'Disease': unique_diseases,
            #'Description': [description[description['Disease'] == disease]['Description'].values[0] 
                            #for disease in unique_diseases]})
        
        # Display table with disease info
        #st.dataframe(
            #disease_df, 
            #column_config={
                #"Disease": st.column_config.TextColumn("Disease Name"),
                #"Description": st.column_config.TextColumn("Short Description", width="large")
            #},
            #hide_index=True,
            #use_container_width=True)

# Run the app
if __name__ == "__main__":
    main()
