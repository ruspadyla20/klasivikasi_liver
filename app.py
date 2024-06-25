import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.metrics import classification_report_imbalanced

# Judul Aplikasi
st.title("Aplikasi Prediksi Penyakit Liver")

# Upload file CSV
uploaded_file = st.file_uploader("Upload file CSV", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write(data)
    
    # Menampilkan nama-nama kolom untuk verifikasi
    st.write("Kolom dalam dataset:", data.columns)

    # Memastikan kolom target yang benar digunakan
    if 'Dataset' in data.columns:
        X = data.drop(columns='Dataset')  # Ganti 'Dataset' dengan nama kolom target yang sesuai
        y = data['Dataset']  # Ganti 'Dataset' dengan nama kolom target yang sesuai
        
        # Inisialisasi dan transformasi menggunakan scaler
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        st.write("Preprocessing selesai. Data siap digunakan.")
    else:
        st.error("Kolom 'Dataset' tidak ditemukan dalam dataset. Mohon periksa nama kolom.")
else:
    st.error("Mohon upload file CSV.")

# Train-test split (jika data sudah tersedia dan 'Dataset' ada dalam kolom)
if 'Dataset' in data.columns:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Inisialisasi dan latih model KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    # Prediksi (misalnya menggunakan data input pengguna)
    prediction = knn.predict(X_test)

    # Tampilkan hasil prediksi
    st.subheader('Hasil Prediksi')
    st.write('Penyakit Liver' if prediction[0] == 1 else 'Tidak Ada Penyakit Liver')

    # Evaluasi model
    st.subheader('Evaluasi Model')

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
