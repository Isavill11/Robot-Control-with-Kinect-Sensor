import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
import pickle as pkl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense, Dropout, BatchNormalization
from keras._tf_keras.keras.callbacks import EarlyStopping
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.metrics import categorical_accuracy
from sklearn.metrics import classification_report

# Load dataset
working_dir = os.getcwd()
df = pd.read_csv(working_dir + '/hand_xyz_coordinates.csv')

# Drop non-numeric columns
df.drop(columns=['source_file', 'source_directory'], inplace=True, errors='ignore')

# Ensure column names are valid
df.columns = [col.replace('.', '_') for col in df.columns]

# Drop NaN values
df.dropna(inplace=True)
# Encode labels
label_encoder = LabelEncoder()
df['gesture_class'] = label_encoder.fit_transform(df['gesture_class'])
num_classes = len(label_encoder.classes_)

# Features and labels
X = df.drop(columns='gesture_class', axis=1)
y = df['gesture_class']

# Ensure all features are numeric
X = X.apply(pd.to_numeric, errors='coerce')

# Split dataset
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42, shuffle=True)
# Normalize features
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)


# Build model
model = Sequential([
    Dense(128, input_dim=X_train.shape[1], activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax')  # Output layer
])

model.summary()


# Compile model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Early stopping
early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, mode='max', restore_best_weights=True)

# Train model
history = model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_val, y_val), callbacks=[early_stopping])

# Predictions
y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)

# Correct classification report
print(classification_report(y_val, y_pred_classes, target_names=label_encoder.classes_))


model.save('gesture_model2.h5')
