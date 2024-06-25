
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import pickle

# Title of the Streamlit application
st.title("Liver Disease Prediction")

# Sidebar for user input
st.sidebar.header("Input Parameters")

def user_input_features():
    age = st.sidebar.number_input("Age", 20, 100, 50)
    gender = st.sidebar.selectbox("Gender", (0, 1))
    totbil = st.sidebar.number_input("Total Bilirubin", 0.0, 100.0, 1.0)
    dirbil = st.sidebar.number_input("Direct Bilirubin", 0.0, 10.0, 0.5)
    alpho = st.sidebar.number_input("Alkaline Phosphotase", 50, 3000, 100)
    alamino = st.sidebar.number_input("Alamine Aminotransferase", 0, 3000, 40)
    asparami = st.sidebar.number_input("Aspartate Aminotransferase", 0, 3000, 40)
    totalpro = st.sidebar.number_input("Total Proteins", 0.0, 10.0, 7.0)
    albumin = st.sidebar.number_input("Albumin", 0.0, 6.0, 3.0)
    Agr = st.sidebar.number_input("Albumin and Globulin Ratio", 0.0, 3.0, 1.0)
    
    data = {
        'Age': age,
        'Gender': gender,
        'Total_Bilirubin': totbil,
        'Direct_Bilirubin': dirbil,
        'Alkaline_Phosphotase': alpho,
        'Alamine_Aminotransferase': alamino,
        'Aspartate_Aminotransferase': asparami,
        'Total_Protiens': totalpro,
        'Albumin': albumin,
        'Albumin_and_Globulin_Ratio': Agr
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Display user input
st.subheader('User Input Parameters')
st.write(input_df)

# Load dataset
data = pd.read_csv('/content/drive/MyDrive/ML 6/Data Pasien penyakit liver.csv')  # Ensure the dataset path is correct

# Preprocessing
X = data.drop(columns='Dataset')  # Replace 'Dataset' with the actual target column name
y = data['Dataset']  # Replace 'Dataset' with the actual target column name

# Standardize data
scaler = StandardScaler()
X = scaler.fit_transform(X)
input_df = scaler.transform(input_df)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Prediction
prediction = knn.predict(input_df)

# Display prediction
st.subheader('Prediction')
st.write('Liver Disease' if prediction[0] == 1 else 'No Liver Disease')

# Evaluate model
st.subheader('Model Evaluation')

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, knn.predict(X_test))
st.write("Confusion Matrix:", conf_matrix)

# Classification Report
class_report = classification_report_imbalanced(y_test, knn.predict(X_test))
st.write("Classification Report:", class_report)

# Accuracy Score
accuracy = accuracy_score(y_test, knn.predict(X_test))
st.write(f"Accuracy Score: {accuracy}")
