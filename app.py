import streamlit as st
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Judul Aplikasi
st.title("Aplikasi Prediksi Penyakit Liver")

# Input Data
st.sidebar.header("Input Parameter")
def user_input_features():
    Age = st.sidebar.number_input("Age 20-100", 20, 100, 50)
    Gender = st.sidebar.selectbox("Gender (0 = Female, 1 = Male)", (0, 1))
    Total_Bilirubin = st.sidebar.number_input("Total Bilirubin", 0.1, 8.0, 1.0)
    Direct_Bilirubin = st.sidebar.number_input("Direct Bilirubin", 0.1, 8.0, 0.5)
    Alkaline_Phosphotase = st.sidebar.number_input("Alkaline Phosphotase", 70, 3000, 200)
    Alamine_Aminotransferase = st.sidebar.number_input("Alamine Aminotransferase", 10, 2000, 100)
    Aspartate_Aminotransferase = st.sidebar.number_input("Aspartate Aminotransferase", 10, 3000, 100)
    Total_Protiens = st.sidebar.number_input("Total Protiens", 2.0, 10.0, 6.0)
    Albumin = st.sidebar.number_input("Albumin", 2.0, 6.0, 4.0)
    Albumin_and_Globulin_Ratio = st.sidebar.number_input("Albumin and Globulin Ratio", 0.1, 2.5, 1.0)
    
    data = {
        'Age': Age,
        'Gender': Gender,
        'Total_Bilirubin': Total_Bilirubin,
        'Direct_Bilirubin': Direct_Bilirubin,
        'Alkaline_Phosphotase': Alkaline_Phosphotase,
        'Alamine_Aminotransferase': Alamine_Aminotransferase,
        'Aspartate_Aminotransferase': Aspartate_Aminotransferase,
        'Total_Protiens': Total_Protiens,
        'Albumin': Albumin,
        'Albumin_and_Globulin_Ratio': Albumin_and_Globulin_Ratio
    }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Tampilkan input pengguna
st.subheader('Input Parameters')
st.write(df)

# Load dataset
liver_data = pd.read_csv('Data Pasien penyakit liver.csv')  # Ganti dengan nama file dataset penyakit liver Anda

# Preprocessing
X = liver_data.drop(columns='Dataset')
y = liver_data['Dataset']

# Standardisasi data
scaler = StandardScaler()
X = scaler.fit_transform(X)
df = scaler.transform(df)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inisialisasi dan latih model KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Prediksi
prediction = knn.predict(df)

# Tampilkan hasil prediksi
st.subheader('Hasil Prediksi')
st.write('Penyakit Liver' if prediction[0] == 1 else 'Tidak Ada Penyakit Liver')
