import streamlit as st
import pandas as pd
import pickle

# Function to load the trained model
@st.cache(allow_output_mutation=True)
def load_model():
    # Load your trained model here
    model = pickle.load(open('randommodel.pkl', 'rb'))  # Replace 'randommodel.pkl' with your pickle model file path
    return model

# Function to preprocess categorical features
def preprocess_categorical(data):
    categorical_cols = ['red_blood_cells', 'pus_cell', 'pus_cell_clumps', 'bacteria',
                        'hypertension', 'diabetes_mellitus', 'coronary_artery_disease',
                        'appetite', 'pedal_edema', 'anemia']
    
    for col in categorical_cols:
        if col == 'appetite':
            data[col] = data[col].map({'good': 0, 'poor': 1})
        elif col == 'red_blood_cells':
            data[col] = data[col].map({'normal': 1, 'abnormal': 0})
        elif col == 'pus_cell':
            data[col] = data[col].map({'normal': 1, 'abnormal': 0})
        elif col == 'pus_cell_clumps':
            data[col] = data[col].map({'notpresent': 0, 'present': 1})
        elif col == 'bacteria':
            data[col] = data[col].map({'notpresent': 0, 'present': 1})
        elif col in ['hypertension', 'diabetes_mellitus']:
            data[col] = data[col].map({'no': 0, 'yes': 1})
        elif col == ['pedal_edema','anemia']:
            data[col] = data[col].map({'no': 0, 'yes': 1})
        elif col == 'coronary_artery_disease':
            data[col] = data[col].map({'no': 0, 'yes': 1})
        else:
            data[col] = data[col].map({'no': 0, 'yes': 1})
    
    return data


# Main function to run the web app
def main():
    st.title('Kidney Disease Prediction')

    # Load the trained model
    model = load_model()

    # Sidebar for user input
    st.sidebar.title('Input Features')

    age = st.sidebar.slider('Age', min_value=1, max_value=100, value=50)
    blood_pressure = st.sidebar.slider('Blood Pressure', min_value=50, max_value=200, value=120)
    specific_gravity = st.sidebar.slider('Specific Gravity', min_value=1.0, max_value=2.0, value=1.01, step=0.01)
    albumin = st.sidebar.slider('Albumin', min_value=0, max_value=5, value=0)
    sugar = st.sidebar.slider('Sugar', min_value=0, max_value=5, value=0)
    red_blood_cell_count = st.sidebar.slider('Red Blood Cell Count', min_value=1.0, max_value=10.0, value=5.5, step=0.1, key='red_blood_cell_count')

    pus_cell = st.sidebar.radio('Pus Cell', ('normal', 'abnormal'))
    pus_cell_clumps = st.sidebar.radio('Pus Cell Clumps', ('present', 'notpresent'))
    bacteria = st.sidebar.radio('Bacteria', ('present', 'notpresent'))
    blood_glucose_random = st.sidebar.slider('Blood Glucose Random', min_value=1, max_value=500, value=100)
    blood_urea = st.sidebar.slider('Blood Urea', min_value=1, max_value=300, value=50)
    serum_creatinine = st.sidebar.slider('Serum Creatinine', min_value=0.1, max_value=15.0, value=1.0, step=0.1)
    sodium = st.sidebar.slider('Sodium', min_value=1, max_value=200, value=135)
    potassium = st.sidebar.slider('Potassium', min_value=1, max_value=200, value=4)
    haemoglobin = st.sidebar.slider('Haemoglobin', min_value=1, max_value=20, value=15)
    packed_cell_volume = st.sidebar.slider('Packed Cell Volume', min_value=1, max_value=60, value=40)
    white_blood_cell_count = st.sidebar.slider('White Blood Cell Count', min_value=1, max_value=20000, value=9000)
    hypertension = st.sidebar.radio('Hypertension', ('yes', 'no'))
    diabetes_mellitus = st.sidebar.radio('Diabetes Mellitus', ('yes', 'no'))
    coronary_artery_disease = st.sidebar.radio('Coronary Artery Disease', ('yes', 'no'))
    appetite = st.sidebar.radio('Appetite', ('good', 'poor'))
    pedal_edema = st.sidebar.radio('Pedal Edema', ('yes', 'no'))
    anemia = st.sidebar.radio('Anemia', ('yes', 'no'))

    # Predict button
    if st.sidebar.button('Predict'):
        # Create input DataFrame
        input_data = pd.DataFrame({'age': [age], 'blood_pressure': [blood_pressure], 'specific_gravity': [specific_gravity],
                                   'albumin': [albumin], 'sugar': [sugar], 'red_blood_cells': ['abnormal' if red_blood_cell_count > 5.0 else 'normal'],
                                   'pus_cell': [pus_cell], 'pus_cell_clumps': [pus_cell_clumps], 'bacteria': [bacteria],
                                   'blood_glucose_random': [blood_glucose_random], 'blood_urea': [blood_urea],
                                   'serum_creatinine': [serum_creatinine], 'sodium': [sodium], 'potassium': [potassium],
                                   'haemoglobin': [haemoglobin], 'packed_cell_volume': [packed_cell_volume],
                                   'white_blood_cell_count': [white_blood_cell_count], 'red_blood_cell_count': [red_blood_cell_count],
                                   'hypertension': [hypertension], 'diabetes_mellitus': [diabetes_mellitus],
                                   'coronary_artery_disease': [coronary_artery_disease], 'appetite': [appetite],
                                   'pedal_edema': [pedal_edema], 'anemia': [anemia]})
        
        # Preprocess categorical features
        input_data = preprocess_categorical(input_data)

        # Make prediction
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)[:, 1]

        # Display prediction result
        st.subheader('Prediction:')
        if prediction[0] == 0:
            st.write('The model predicts that the patient has kidney disease.')
        else:
            st.write('The model predicts that the patient does not have kidney disease.')

        # Display prediction probability
        st.subheader('Prediction Probability:')
        if prediction[0] == 0:
            st.write(f'The probability of having kidney disease is {prediction_proba[0]*100:.2f}%.')
        else:
            st.write(f'The probability of not having kidney disease is {prediction_proba[0]*100:.2f}%.')
     

if __name__ == '__main__':
    main()
