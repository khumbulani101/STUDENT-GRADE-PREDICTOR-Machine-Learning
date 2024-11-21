import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle

# Load dataset
data = pd.read_csv('student-mat.csv', sep=';')

# Select relevant features
data = data[['studytime', 'absences', 'G1', 'G3']]  # G3 is the final grade

# Handle missing values
data.dropna(inplace=True)

# Define features (X) and target (y)
X = data[['studytime', 'absences', 'G1']]
y = data['G3']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")

# Save the trained model
with open('grade_predictor.pkl', 'wb') as f:
    pickle.dump(model, f)

   
