import os
from operator import indexOf

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
import pickle as pkl
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense, Dropout, BatchNormalization
from keras._tf_keras.keras.callbacks import EarlyStopping
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.metrics import categorical_accuracy
from sklearn.metrics import classification_report

working_dir = os.getcwd()
name = 'closer_data_coords'
df = pd.read_csv(working_dir + '/' + name + '.csv')

df.drop(columns=['source_file', 'source_directory'], inplace=True, errors='ignore')

df.columns = [col.replace('.', '_') for col in df.columns]
df_dropping = [col for col in df.columns if '_z' in col]
# df_class_remove = ['C-sign', 'palm_up', 'thumbs_down', 'thumbs_up']


df.dropna(inplace=True)
for name in df_dropping:
    df = df.drop(columns = name)

# for item in df_class_remove:
#     df = df[df['gesture_class'] != item]

# Encode labels
label_encoder = LabelEncoder()
df['gesture_class'] = label_encoder.fit_transform(df['gesture_class'])
num_classes = len(label_encoder.classes_)

# print(df['gesture_class'])
X = df.drop(columns='gesture_class', axis=1)
y = df['gesture_class']
print(y.count)
X = X.apply(pd.to_numeric, errors='coerce')


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42, shuffle=True)

# scaler = MinMaxScaler()
# X_train = scaler.fit_transform(X_train)
# X_val = scaler.transform(X_val)

print(f'X train shape: {X_train.shape}')


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

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, mode='max', restore_best_weights=True)

#train
history = model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_val, y_val), callbacks=[early_stopping])

#analyze accuracy
y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)

#class report
print(classification_report(y_val, y_pred_classes, target_names=label_encoder.classes_))


model.save(name+'_model.keras')
# joblib.dump(scaler, name+'_scaler.pkl')
#