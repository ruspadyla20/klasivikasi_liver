import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open('klasifikasi_liver.pkl', 'rb') as file :
model = pickle.load(file)

# Define the prediction function
def predict_species(Age, Gender, Total_Bilirubin, Direct_Bilirubin,	Alkaline_Phosphotase,	Alamine_Aminotransferase,	Aspartate_Aminotransferase,	Total_Protiens,	Albumin,	Albumin_and_Globulin_Ratio):
    # Create numpy array from user input
    input_data = np.array([[Age, Gender, Total_Bilirubin, Direct_Bilirubin,	Alkaline_Phosphotase,	Alamine_Aminotransferase,	Aspartate_Aminotransferase,	Total_Protiens,	Albumin,	Albumin_and_Globulin_Ratio]])

    # Perform prediction using the loaded model
    prediction = model.predict(input_data)

    # Return the predicted species
    return prediction[0]

# Streamlit app
def main():
    # Set title and description
    st.title('Klasifikasi Penyakit Liver')
    

    # Input fields for user to enter values
    Age = st.number_input('Age')
    Gender = st.number_input('Gender')
    Total_Bilirubin = st.number_input('Total_Bilirubin')
    Direct_Bilirubin = st.number_input('Direct_Bilirubin')
    Alkaline_Phosphotase = st.number_input('Alkaline_Phosphotase')
    Alamine_Aminotransferase = st.number_input('Alamine_Aminotransferase')
    Aspartate_Aminotransferase = st.number_input('Aspartate_Aminotransferase')
    Total_Protiens = st.number_input('Total_Protiens')
    Albumin = st.number_input('Albumin')
    Albumin_and_Globulin_Ratio = st.number_input('Albumin_and_Globulin_Ratio')

    # When 'Predict' button is clicked, make prediction and display result
    if st.button('Predict'):
        prediction = predict_species(Age, Gender, Total_Bilirubin, Direct_Bilirubin,	Alkaline_Phosphotase,	Alamine_Aminotransferase,	Aspartate_Aminotransferase,	Total_Protiens,	Albumin,	Albumin_and_Globulin_Ratio)
        st.write(f'Predicted Species: {prediction}')

if __name__ == '__main__':
    main()
