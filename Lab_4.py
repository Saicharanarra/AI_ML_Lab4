from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import pandas as pd

# Load dataset
df = pd.read_csv("F:/Saicharan_AI_Sem2/AI In Enterprise Systems/lab4_v1/Fish.csv")

# Display first few rows
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Encode species column(convert categorical column to numerical)
le = LabelEncoder()
df["Species"] = le.fit_transform(df["Species"])

# Define Features (X) and Target Variable (y)
X = df.drop(columns=["Weight"])  # Features
y = df["Weight"]  # Target Variable

# Split Data for Training and Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate performance
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Save the model
joblib.dump(model, "fish_model.pkl")
print("Model saved successfully!")




