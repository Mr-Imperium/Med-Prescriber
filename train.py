import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle

# Load the dataset
dataset = pd.read_csv('data/Training.csv')

# Prepare the data
X = dataset.drop('prognosis', axis=1)
y = dataset['prognosis']

# Encoding prognosis
le = LabelEncoder()
le.fit(y)
Y = le.transform(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=20)

# Train SVC model
svc = SVC(kernel='linear')
svc.fit(X_train, y_train)

# Save the model
pickle.dump(svc, open('svc.pkl', 'wb'))

# Load additional datasets
sym_des = pd.read_csv("data/symtoms_df.csv")
precautions = pd.read_csv("data/precautions_df.csv")
workout = pd.read_csv("data/workout_df.csv")
description = pd.read_csv("data/description.csv")
medications = pd.read_csv('data/medications.csv')
diets = pd.read_csv("data/diets.csv")

# Create symptoms and diseases dictionaries
symptoms_dict = {symptom: index for index, symptom in enumerate(X.columns)}
diseases_list = {disease: index for index, disease in enumerate(le.classes_)}

def helper(dis):
    # Handle description
    desc_df = description[description['Disease'] == dis]
    desc = " ".join(desc_df['Description']) if not desc_df.empty else "No description available."

    # Handle precautions
    pre_df = precautions[precautions['Disease'] == dis]
    pre = []
    if not pre_df.empty:
        pre = [pre_df['Precaution_1'].values[0], 
               pre_df['Precaution_2'].values[0], 
               pre_df['Precaution_3'].values[0], 
               pre_df['Precaution_4'].values[0]]
        # Remove any None or empty values
        pre = [p for p in pre if p and str(p).strip()]

    # Handle medications
    med_df = medications[medications['Disease'] == dis]
    med = med_df['Medication'].tolist() if not med_df.empty else []

    # Handle diets
    die_df = diets[diets['Disease'] == dis]
    die = die_df['Diet'].tolist() if not die_df.empty else []

    # Handle workout
    wrkout_df = workout[workout['disease'] == dis]
    wrkout = wrkout_df['workout'].tolist() if not wrkout_df.empty else []

    return desc, pre, med, die, wrkout

def get_predicted_value(patient_symptoms):
    # Create input vector based on all possible symptoms
    input_vector = np.zeros(len(symptoms_dict))
    
    # Check if all input symptoms are valid
    invalid_symptoms = [sym for sym in patient_symptoms if sym not in symptoms_dict]
    if invalid_symptoms:
        print(f"Warning: Invalid symptoms: {invalid_symptoms}")
        return None
    
    # Mark input symptoms as present
    for item in patient_symptoms:
        input_vector[symptoms_dict[item]] = 1
    
    # Predict and map back to disease name
    predicted_index = svc.predict([input_vector])[0]
    
    # Reverse the label encoding to get the disease name
    predicted_disease = le.inverse_transform([predicted_index])[0]
    
    return predicted_disease

# Main prediction loop
while True:
    # Get user symptoms
    symptoms = input("Enter your symptoms (comma-separated, or 'quit' to exit): ")
    
    # Check for quit condition
    if symptoms.lower() == 'quit':
        break
    
    # Process symptoms
    user_symptoms = [s.strip() for s in symptoms.split(',')]
    user_symptoms = [symptom.strip("[]' ") for symptom in user_symptoms]
    
    # Predict disease
    predicted_disease = get_predicted_value(user_symptoms)
    
    if predicted_disease:
        # Get additional information
        desc, pre, med, die, wrkout = helper(predicted_disease)

        # Print results
        print("\n=================Predicted Disease============")
        print(predicted_disease)
        print("\n=================Description==================")
        print(desc)
        
        # Print Precautions with error handling
        print("\n=================Precautions==================")
        if pre:
            for i, p_i in enumerate(pre, 1):
                print(f"{i}: {p_i}")
        else:
            print("No precautions available.")

        # Print Medications with error handling
        print("\n=================Medications==================")
        if med:
            for i, m_i in enumerate(med, 1):
                print(f"{i}: {m_i}")
        else:
            print("No medications information available.")

        # Print Workout with error handling
        print("\n=================Non-Pharmacological Measures==================")
        if wrkout:
            for i, w_i in enumerate(wrkout, 1):
                print(f"{i}: {w_i}")
        else:
            print("No workout information available.")

        # Print Diets with error handling
        print("\n=================Diets==================")
        if die:
            for i, d_i in enumerate(die, 1):
                print(f"{i}: {d_i}")
        else:
            print("No diet information available.")
    else:
        print("Unable to predict disease. Please check your symptoms.")