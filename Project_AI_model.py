import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import cohen_kappa_score
from imblearn.over_sampling import SMOTE
 

df = pd.read_csv('wine.csv')
print(df.info())
print(df.head())

#To display bar chart to check if the dataset is balanced or not
plt.figure(figsize=(8, 5))
sns.countplot(x='quality', data=df, palette='Set2')
plt.title('Distribution of wine Quality')
plt.xlabel('Quality')
plt.ylabel('Count')
plt.show()

wine_data = pd.read_csv('wine.csv')

plt.figure(figsize=(12, 8))
#To generate heat map 
sns.heatmap(wine_data.corr(), annot=True, cmap='coolwarm')
plt.title('Heatmap of Wine Quality Correlations')
plt.show()


# Load the dataset
wine_data = pd.read_csv('wine.csv')

# Encode the quality labels (good/bad) into binary values
label_encoder = LabelEncoder()
wine_data['quality'] = label_encoder.fit_transform(wine_data['quality'])#transformation data transformation

# Separate features and target variable
X = wine_data.drop('quality', axis=1)
y = wine_data['quality']

# class balancing 
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) 

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Build the MLP model
# using batch normilization.
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(BatchNormalization())
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1, activation='sigmoid'))  # Use 'softmax' if you have multiple classes

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) #loss function calculates the error 
# Train the model
history = model.fit(X_train, y_train, epochs=25, batch_size=16, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy:.2f}%')
# Make predictions
y_pred = (model.predict(X_test) > 0.5).astype("int32")  # Convert probabilities to binary
kappa = cohen_kappa_score(y_test, y_pred)
print(f'kappa score: {kappa:.2f}')
# Print classification report and confusion matrix
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

plt.figure(figsize=(8, 6))
plt.title('Accuracy')
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
# to count the number of Good and bad samples in the dataset
wine_data = pd.read_csv('wine.csv') 

quality_counts = wine_data['quality'].value_counts()

print("Counts of quality categories:")
print(quality_counts)
