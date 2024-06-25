import pickle
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Judul Aplikasi
st.title("Data Pasien Penyakit Liver")

# Input Data
st.sidebar.header("Input Parameter")
def user_input_features():
    age = st.sidebar.number_input("Age", 20, 100, 50)
    gender = st.sidebar.selectbox("Gender", (0, 1))
    totbil = st.sidebar.selectbox("Total_Bilirubin", (0, 1, 2, 3))
    dirbil = st.sidebar.number_input("Direct_Bilirubin", 80, 200, 120)
    alpho = st.sidebar.number_input("Alkaline_Phosphotase", 100, 400, 200)
    alamino = st.sidebar.selectbox("Alamine_Aminotransferase", (0, 1))
    asparami = st.sidebar.selectbox("Aspartate_Aminotransferase", (0, 1, 2))
    totalpro = st.sidebar.number_input("Total_Protiens", 70, 210, 150)
    albumin = st.sidebar.selectbox("Albumin", (0, 1))
    Agr = st.sidebar.number_input("SAlbumin_and_Globulin_Ratio", 0.0, 6.0, 1.0)
    
    data = {
        'age': age,
        'gender': gender,
        'totbil': totbil,
        'dirbil': dirbil,
        'alpho': alpho,
        'alamino': alamino,
        'asparami': asparami,
        'totalpro': totalpro,
        'albumin': albumin,
        'Agr': Agr,
    }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Tampilkan input pengguna
st.subheader('Input Parameters')
st.write(df)

# Load dataset
data = pd.read_csv('Data Pasien penyakit liver.csv')  # Pastikan file dataset tersedia

# Preprocessing
X = data.drop(columns='target')
y = data['target']

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
