import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
from os import system


# Step 1: Load the dataset
data = pd.read_csv('datasets/iot_telemetry_data.csv')  # Replace with the actual path to your dataset

# Step 2: Filter data for the specific device
device_data = data[data['device'] == 'b8:27:eb:bf:9d:51']

# Step 3: Create synthetic labels
device_data['fire'] = (
    (device_data['co'] > 0.02) |
    (device_data['humidity'] < 30) |
    (device_data['lpg'] > 0.01) |
    (device_data['smoke'] > 0.03) |
    (device_data['temp'] > 100)
).astype(int)

# Step 4: Select features and labels
X = device_data[['co', 'humidity', 'lpg', 'smoke', 'temp']]
y = device_data['fire']

# Step 5: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 7: Evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Step 8: Save the trained model
joblib.dump(model, 'models/sensor_fire_model.pkl')