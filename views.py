

import tkinter as tk
from tkinter import messagebox
from tensorflow.keras.models import load_model
import pandas as pd
import joblib  # For loading the saved scaler

# Load the trained model
model = load_model(r"D:\Data\AI & ML\loan_approval_model.h5")

# Load the saved scaler
scaler = joblib.load(r"D:\Data\AI & ML\scaler.pkl")

# Define the column names as they were during training
columns = [
    'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
    'Property_Area', 'ApplicantIncome', 'CoapplicantIncome',
    'LoanAmount', 'Loan_Amount_Term', 'Credit_History'
]
print(columns)
for i in columns:
    print(i)

# Create the main window
root = tk.Tk()
root.title("Loan Approval Prediction")

# Define the input fields and labels
fields = {
    "Gender": ["Male", "Female"],
    "Married": ["Yes", "No"],
    "Dependents": ["0", "1", "2", "3+"],
    "Education": ["Graduate", "Not Graduate"],
    "Self_Employed": ["Yes", "No"],
    "Property_Area": ["Rural", "Semiurban", "Urban"],
    "ApplicantIncome": "numeric",
    "CoapplicantIncome": "numeric",
    "LoanAmount": "numeric",
    "Loan_Amount_Term": "numeric",
    "Credit_History": "numeric",
}

inputs = {}

for i, (field, values) in enumerate(fields.items()):
    tk.Label(root, text=field).grid(row=i, column=0)
    if values == "numeric":
        inputs[field] = tk.Entry(root)
        inputs[field].grid(row=i, column=1)
    else:
        inputs[field] = tk.StringVar(root)
        tk.OptionMenu(root, inputs[field], *values).grid(row=i, column=1)
        inputs[field].set(values[0])

def predict_loan_status():
    # Prepare the input data
    data = {}
    for field in fields:
        if fields[field] == "numeric":
            data[field] = float(inputs[field].get())
        else:
            data[field] = inputs[field].get()

    # Create a DataFrame with the same columns as used for fitting
    df = pd.DataFrame([data], columns=columns)

    # Map categorical values
    df['Dependents'] = df['Dependents'].replace({"3+": 4})
    df['Married'] = df['Married'].replace({"Yes": 1, "No": 0})
    df['Gender'] = df['Gender'].replace({"Male": 1, "Female": 0})
    df['Self_Employed'] = df['Self_Employed'].replace({"Yes": 1, "No": 0})
    df['Property_Area'] = df['Property_Area'].replace({"Rural": 0, "Semiurban": 1, "Urban": 2})
    df['Education'] = df['Education'].replace({"Graduate": 1, "Not Graduate": 0})

    # Ensure the DataFrame has the correct column order
    df = df[columns]

    # Scale the data
    df_scaled = scaler.transform(df)

    # Predict
    prediction = model.predict(df_scaled)
    loan_status = "Approved" if prediction[0][0] > 0.5 else "Not Approved"

    messagebox.showinfo("Loan Status", f"The loan status is: {loan_status}")

# Add a button to predict the loan status
tk.Button(root, text="Predict Loan Status", command=predict_loan_status).grid(row=len(fields), column=0, columnspan=2)

# Run the application
root.mainloop()



# tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, GaussianNoise
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


scaler = MinMaxScaler()

path = r"D:\Data\AI & ML\train_u6lujuX_CVtuZ9i.csv"
data = pd.read_csv(path)

data = data.dropna()
data['Loan_Status'] = data['Loan_Status'].replace({"Y": 1, "N": 0})
data['Dependents'] = data['Dependents'].replace({"3+": 4})
data['Married'] = data['Married'].replace({"Yes": 1, "No": 0})
data['Gender'] = data['Gender'].replace({"Male": 1, "Female": 0})
data['Self_Employed'] = data['Self_Employed'].replace({"Yes": 1, "No": 0})
data['Property_Area'] = data['Property_Area'].replace({"Rural": 0, "Semiurban": 1, "Urban": 2})
data['Education'] = data['Education'].replace({"Graduate": 1, "Not Graduate": 0})


y = data['Loan_Status']
X = data.drop(['Loan_Status', 'Loan_ID'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=2)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential()
model.add(Dense(units=200, activation='relu', input_dim=X_train.shape[1]))
model.add(BatchNormalization())
model.add(Dense(units=100, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(units=50, activation='relu'))
# model.add(Dropout(rate=0.3))
model.add(Dense(units=40, activation='relu'))
model.add(GaussianNoise(stddev=0.99))
model.add(Dense(units=30, activation='relu'))

# Binary output layer
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=30, epochs=100, validation_data=(X_test, y_test), verbose=1)

model.save('loan_approval_model2.h5')

loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')
print(f'Test Accuracy: {accuracy}')



import joblib
import numpy as np

from tensorflow.keras.models import load_model
loaded_model = load_model(r"D:\Data\AI & ML\loan_approval_model.h5")
scaler = joblib.load(r"D:\Data\AI & ML\scaler.pkl")

new_data = np.array([[1, 0, 2, 1, 0, 5000, 0, 150, 360, 1, 2]])

new_data_scaled = scaler.transform(new_data)

prediction = loaded_model.predict(new_data_scaled)
prediction_binary = (prediction > 0.5).astype(int)

print(f'Prediction (Probability): {prediction[0][0]}')
print(f'Prediction (Class): {prediction_binary[0][0]}')


# GUI App

import joblib
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from tensorflow.keras.models import load_model

# Load the model and scaler
loaded_model = load_model(r"D:\Data\AI & ML\loan_approval_model.h5")
scaler = joblib.load(r"D:\Data\AI & ML\scaler.pkl")

# Function to make a prediction
def make_prediction():
    try:
        # Get data from the GUI input fields
        gender = int(gender_var.get())
        married = int(married_var.get())
        dependents = int(dependents_var.get())
        education = int(education_var.get())
        self_employed = int(self_employed_var.get())
        applicant_income = float(entry_applicant_income.get())
        coapplicant_income = float(entry_coapplicant_income.get())
        loan_amount = float(entry_loan_amount.get())
        loan_amount_term = float(entry_loan_amount_term.get())
        credit_history = int(credit_history_var.get())
        property_area = int(property_area_var.get())

        # Create a numpy array from the input
        new_data = np.array([[gender, married, dependents, education, self_employed,
                              applicant_income, coapplicant_income, loan_amount,
                              loan_amount_term, credit_history, property_area]])

        # Scale the data
        new_data_scaled = scaler.transform(new_data)

        # Make prediction
        prediction = loaded_model.predict(new_data_scaled)
        prediction_binary = (prediction > 0.5).astype(int)

        # Show the prediction result
        probability = prediction[0][0]
        result_class = prediction_binary[0][0]
        result_text = f"Prediction (Probability): {probability:.2f}\nPrediction (Class): {'Approved' if result_class == 1 else 'Rejected'}"
        messagebox.showinfo("Prediction Result", result_text)

    except ValueError as ve:
        messagebox.showerror("Input Error", f"Invalid input: {ve}")

# Create the main window
root = tk.Tk()
root.title("Loan Approval Prediction")
root.geometry("400x500")
root.resizable(False, False)

# Apply a theme
style = ttk.Style(root)
style.theme_use('clam')

# Create and place the input fields with labels
frame = ttk.Frame(root, padding="10 10 10 10")
frame.pack(fill=tk.BOTH, expand=True)

# Gender
ttk.Label(frame, text="Gender:").grid(row=0, column=0, sticky='e', pady=5)
gender_var = tk.StringVar()
gender_combobox = ttk.Combobox(frame, textvariable=gender_var, values=["1", "0"], state='readonly')
gender_combobox.grid(row=0, column=1, pady=5)
gender_combobox.current(0)

# Married
ttk.Label(frame, text="Married:").grid(row=1, column=0, sticky='e', pady=5)
married_var = tk.StringVar()
married_combobox = ttk.Combobox(frame, textvariable=married_var, values=["1", "0"], state='readonly')
married_combobox.grid(row=1, column=1, pady=5)
married_combobox.current(0)

# Dependents
ttk.Label(frame, text="Dependents:").grid(row=2, column=0, sticky='e', pady=5)
dependents_var = tk.StringVar()
dependents_combobox = ttk.Combobox(frame, textvariable=dependents_var, values=["0", "1", "2", "3+"], state='readonly')
dependents_combobox.grid(row=2, column=1, pady=5)
dependents_combobox.current(0)

# Education
ttk.Label(frame, text="Education:").grid(row=3, column=0, sticky='e', pady=5)
education_var = tk.StringVar()
education_combobox = ttk.Combobox(frame, textvariable=education_var, values=["1", "0"], state='readonly')
education_combobox.grid(row=3, column=1, pady=5)
education_combobox.current(0)

# Self Employed
ttk.Label(frame, text="Self Employed:").grid(row=4, column=0, sticky='e', pady=5)
self_employed_var = tk.StringVar()
self_employed_combobox = ttk.Combobox(frame, textvariable=self_employed_var, values=["1", "0"], state='readonly')
self_employed_combobox.grid(row=4, column=1, pady=5)
self_employed_combobox.current(0)

# Applicant Income
ttk.Label(frame, text="Applicant Income:").grid(row=5, column=0, sticky='e', pady=5)
entry_applicant_income = ttk.Entry(frame)
entry_applicant_income.grid(row=5, column=1, pady=5)

# Coapplicant Income
ttk.Label(frame, text="Coapplicant Income:").grid(row=6, column=0, sticky='e', pady=5)
entry_coapplicant_income = ttk.Entry(frame)
entry_coapplicant_income.grid(row=6, column=1, pady=5)

# Loan Amount
ttk.Label(frame, text="Loan Amount:").grid(row=7, column=0, sticky='e', pady=5)
entry_loan_amount = ttk.Entry(frame)
entry_loan_amount.grid(row=7, column=1, pady=5)

# Loan Amount Term
ttk.Label(frame, text="Loan Amount Term:").grid(row=8, column=0, sticky='e', pady=5)
entry_loan_amount_term = ttk.Entry(frame)
entry_loan_amount_term.grid(row=8, column=1, pady=5)

# Credit History
ttk.Label(frame, text="Credit History:").grid(row=9, column=0, sticky='e', pady=5)
credit_history_var = tk.StringVar()
credit_history_combobox = ttk.Combobox(frame, textvariable=credit_history_var, values=["1", "0"], state='readonly')
credit_history_combobox.grid(row=9, column=1, pady=5)
credit_history_combobox.current(0)

# Property Area
ttk.Label(frame, text="Property Area:").grid(row=10, column=0, sticky='e', pady=5)
property_area_var = tk.StringVar()
property_area_combobox = ttk.Combobox(frame, textvariable=property_area_var, values=["0", "1", "2"], state='readonly')
property_area_combobox.grid(row=10, column=1, pady=5)
property_area_combobox.current(0)

# Create a button to make a prediction
predict_button = ttk.Button(frame, text="Predict Loan Approval", command=make_prediction)
predict_button.grid(row=11, column=0, columnspan=2, pady=10)

# Start the main loop
root.mainloop()
