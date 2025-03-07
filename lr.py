import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load dataset from CSV
df = pd.read_csv("data_raw.csv")  

# Separate features and labels
y = df.iloc[:, 0].values  
X = df.iloc[:, 1:].values

# Normalize 
X = X / 255.0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1667, random_state=42)  # 50,000 train / 10,000 test


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=300,solver='saga',n_jobs=-1)
#chose saga over lbfgs because I got a little higher accuracy
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = 100*accuracy_score(y_test, y_pred)
print(f"Linear Regression (Logistic Regression) Accuracy: {accuracy:.2f}%")