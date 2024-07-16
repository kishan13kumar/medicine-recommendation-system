import numpy as np
import streamlit as st
from pathlib import Path
import base64

# Function to convert image to bytes
def img_to_bytes(img_path):
    img_path = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_path).decode()
    return encoded

# Function to create the sidebar
def sidebar():
    st.sidebar.markdown(f'''<img src='data:image/png;base64,{img_to_bytes("m.jpg")}' class='img-fluid' width=225 height=280>''', unsafe_allow_html=True)
    st.sidebar.markdown("""
       <div style="padding: 10px;">
           <h2 style="color: #1f77b4;">Allopathy Recommendation System</h2>
           <p style="font-size: 14px; line-height: 1.6;">
               This recommendation system aids healthcare professionals in making informed decisions by analyzing patient data to provide evidence-based diagnosis and treatment suggestions. It enhances clinical decision-making and patient care outcomes, ultimately improving the quality of healthcare delivery.
           </p>
          <p style="font-size: 14px; line-height: 1.6;">
               By leveraging machine learning techniques and clinical expertise, this system empowers healthcare providers to deliver more effective and efficient care, leading to better patient outcomes and increased healthcare efficiency.
           </p>
       </div>
       """, unsafe_allow_html=True)

# Define the allopathic medicines and their associated diseases
medication_diseases = {
    "Aspirin": ["Pain", "Fever", "Inflammation"],
    "Paracetamol": ["Pain", "Fever"],
    "Ibuprofen": ["Pain", "Fever", "Inflammation"],
    "Amoxicillin": ["Bacterial Infections (e.g., Ear Infections, Sinus Infections, Pneumonia)"],
    "Ciprofloxacin": ["Bacterial Infections (e.g., Urinary Tract Infections, Respiratory Infections)"],
    "Metformin": ["Type 2 Diabetes"],
    "Insulin": ["Diabetes (Type 1 and Type 2)"],
    "Atorvastatin": ["High Cholesterol", "Heart Disease"],
    "Lisinopril": ["Hypertension", "Heart Failure"],
    "Amlodipine": ["Hypertension", "Angina"],
    "Hydrochlorothiazide": ["Hypertension", "Edema"],
    "Losartan": ["Hypertension", "Heart Failure"],
    "Omeprazole": ["Gastroesophageal Reflux Disease (GERD)", "Ulcers"],
    "Pantoprazole": ["GERD", "Ulcers"],
    "Ranitidine": ["GERD", "Ulcers"],
    "Simvastatin": ["High Cholesterol", "Heart Disease"],
    "Levothyroxine": ["Hypothyroidism"],
    "Warfarin": ["Blood Clots", "Stroke Prevention"],
    "Clopidogrel": ["Blood Clots", "Heart Attack Prevention"],
    "Metoprolol": ["Hypertension", "Angina", "Heart Failure"],
    "Prednisone": ["Inflammatory Conditions (e.g., Asthma, Rheumatoid Arthritis)"],
    "Cetirizine": ["Allergies", "Hay Fever"],
    "Loratadine": ["Allergies", "Hay Fever"],
    "Furosemide": ["Edema", "Heart Failure"],
    "Albuterol": ["Asthma", "Chronic Obstructive Pulmonary Disease (COPD)"],
    "Sertraline": ["Depression", "Anxiety Disorders"],
    "Fluoxetine": ["Depression", "Anxiety Disorders"],
    "Escitalopram": ["Depression", "Anxiety Disorders"],
    "Paroxetine": ["Depression", "Anxiety Disorders"],
    "Tramadol": ["Pain (Moderate to Severe)"],
    "Morphine": ["Severe Pain", "Palliative Care"],
    "Codeine": ["Mild to Moderate Pain", "Cough"],
    "Oxycodone": ["Moderate to Severe Pain"],
    "Gabapentin": ["Neuropathic Pain", "Seizures"],
    "Pregabalin": ["Neuropathic Pain", "Fibromyalgia"],
    "Aripiprazole": ["Schizophrenia", "Bipolar Disorder"],
    "Quetiapine": ["Schizophrenia", "Bipolar Disorder"],
    "Risperidone": ["Schizophrenia", "Bipolar Disorder"],
    "Alprazolam": ["Anxiety Disorders", "Panic Disorders"],
    "Diazepam": ["Anxiety Disorders", "Muscle Spasms", "Seizures"]
}

unique_diseases = sorted(list(set(disease for diseases in medication_diseases.values() for disease in diseases)))

# Create the medication-disease efficacy matrix
medication_data = []
for medication, diseases in medication_diseases.items():
    efficacy_row = [1 if disease in diseases else 0 for disease in unique_diseases]
    medication_data.append(efficacy_row)

medication_data = np.array(medication_data)

# Define the matrix factorization recommendation class
class MedicationRecommendation:
    def __init__(self, medication_data, num_factors, learning_rate, reg_param, num_epochs):
        self.medication_data = medication_data
        self.num_medications, self.num_diseases = medication_data.shape
        self.num_factors = num_factors
        self.learning_rate = learning_rate
        self.reg_param = reg_param
        self.num_epochs = num_epochs
        self.P = np.random.normal(scale=1./self.num_factors, size=(self.num_medications, self.num_factors))
        self.Q = np.random.normal(scale=1./self.num_factors, size=(self.num_diseases, self.num_factors))

    def train(self):
        for epoch in range(self.num_epochs):
            for i in range(self.num_medications):
                for j in range(self.num_diseases):
                    if self.medication_data[i][j] > 0:
                        eij = self.medication_data[i][j] - np.dot(self.P[i, :], self.Q[j, :].T)
                        self.P[i, :] += self.learning_rate * (eij * self.Q[j, :] - self.reg_param * self.P[i, :])
                        self.Q[j, :] += self.learning_rate * (eij * self.P[i, :] - self.reg_param * self.Q[j, :])
            if epoch % 10 == 0:
                error = self.calculate_error()
                print(f"Epoch {epoch}, Error: {error}")

    def calculate_error(self):
        errors = []
        for i in range(self.num_medications):
            for j in range(self.num_diseases):
                if self.medication_data[i][j] > 0:
                    errors.append((self.medication_data[i][j] - np.dot(self.P[i, :], self.Q[j, :].T)) ** 2)
        return np.mean(errors)

    def get_recommendations(self, diseases, top_n=5):
        disease_indices = [unique_diseases.index(disease) for disease in diseases if disease in unique_diseases]
        if not disease_indices:
            return []

        medication_scores = np.dot(self.P[:, :], self.Q[disease_indices, :].T)
        top_medication_indices = np.argsort(medication_scores, axis=0)[::-1][:top_n]
        return top_medication_indices

# Initialize the medication recommendation system
recommendation_system = MedicationRecommendation(medication_data, num_factors=5, learning_rate=0.01, reg_param=0.01, num_epochs=100)
recommendation_system.train()

# Create Streamlit app
st.title("Allopathy Medicine Recommendation System")

# Sidebar
sidebar()

# Main content
with st.form(key='disease_form'):
    input_diseases = st.text_input("Enter diseases (comma-separated): ").strip().split(',')
    submit_button = st.form_submit_button(label='Get Recommendations')

if submit_button:
    if input_diseases:
        recommendations = recommendation_system.get_recommendations(input_diseases)
        if recommendations:
            for i, disease in enumerate(input_diseases):
                if disease in unique_diseases:
                    st.write(f"Top recommendations for {disease}: {[list(medication_diseases.keys())[j] for j in recommendations[:, i]]}")
                else:
                    st.write(f"No recommendations available for {disease}.")
        else:
            st.write("No recommendations available for the entered diseases.")
    else:
        st.write("Please enter at least one disease.")
