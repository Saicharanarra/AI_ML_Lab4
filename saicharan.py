# Fish Market Model Code (fish_model.py)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle

# Load dataset
df = pd.read_csv("Fish.csv")

# Feature Engineering
df = pd.get_dummies(df, drop_first=True)  # Encoding categorical features
X = df.drop('Weight', axis=1)
y = df['Weight']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
print(f"MSE: {mean_squared_error(y_test, y_pred)}")

# Save Model
with open('fish_model.pkl', 'wb') as file:
    pickle.dump(model, file)
