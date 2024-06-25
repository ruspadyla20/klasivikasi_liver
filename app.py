import streamlit as st
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from imblearn.metrics import classification_report_imbalanced

# Judul Aplikasi
st.title("Aplikasi Prediksi Penyakit Liver dengan KNN")

# Upload file CSV
uploaded_file = st.file_uploader("Upload file CSV", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Data yang diunggah:")
    st.write(data)
else:
    st.stop()

# Memastikan kolom target yang benar digunakan
if 'Dataset' in data.columns:
    X = data.drop(columns=['Dataset'])  # Ganti 'Dataset' dengan nama kolom target yang sesuai
    y = data['Dataset']  # Ganti 'Dataset' dengan nama kolom target yang sesuai
    
    # Inisialisasi dan transformasi menggunakan scaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    st.write("Preprocessing selesai. Data siap digunakan.")
else:
    st.error("Kolom 'Dataset' tidak ditemukan dalam dataset. Mohon periksa nama kolom.")
    st.stop()

# Input data pengguna untuk prediksi
st.header("Masukkan Data untuk Prediksi")
input_data = {}
for column in X.columns:
    input_data[column] = st.number_input(f'Masukkan {column}', step=0.01)

input_df = pd.DataFrame([input_data])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inisialisasi dan latih model KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Prediksi menggunakan data input pengguna
prediction = knn.predict(input_df)

# Tampilkan hasil prediksi
st.header("Hasil Prediksi")
st.write('Prediksi Penyakit Liver:' if prediction[0] == 1 else 'Prediksi Tidak Ada Penyakit Liver')

# Evaluasi model
st.header("Evaluasi Model")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, knn.predict(X_test))
st.write("Confusion Matrix:")
st.write(conf_matrix)

# Classification Report
class_report = classification_report_imbalanced(y_test, knn.predict(X_test))
st.write("Classification Report:")
st.write(class_report)

# Accuracy Score
accuracy = accuracy_score(y_test, knn.predict(X_test))
st.write(f"Accuracy Score: {accuracy}")
