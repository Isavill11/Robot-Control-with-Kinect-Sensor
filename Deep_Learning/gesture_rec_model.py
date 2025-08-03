import os
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense, Dropout, BatchNormalization
from keras._tf_keras.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras._tf_keras.keras.optimizers import Adam
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns



# Load Dataset
working_dir = os.getcwd()
name = 'closer_data_coords'
df = pd.read_csv(os.path.join(working_dir, name + '.csv'))

# Drop unnecessary columns
df.drop(columns=['source_file', 'source_directory'], inplace=True, errors='ignore')

df.columns = [col.replace('.', '_') for col in df.columns]
df_dropping = [col for col in df.columns if '_z' in col]

## need to remove a class?
# df_class_remove = ['C-sign', 'palm_up', 'thumbs_down', 'thumbs_up']
# for item in df_class_remove:
#     df = df[df['gesture_class'] != item]

df.dropna(inplace=True)
for name in df_dropping:
    df = df.drop(columns = name)

GESTURE_CLASSES= df['gesture_class'].tolist()
label_encoder = LabelEncoder()
df['gesture_class'] = label_encoder.fit_transform(df['gesture_class'])
num_classes = len(label_encoder.classes_)

X = df.drop(columns=['gesture_class'])
y = df['gesture_class']
X = X.apply(pd.to_numeric, errors='coerce')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=42, shuffle=True, stratify=y)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

class_counts = df['gesture_class'].value_counts()

print("Class Distribution:\n", class_counts)
plt.figure(figsize=(8, 5))
class_counts.plot(kind='bar', color='c')
plt.xlabel("Gesture Class")
plt.ylabel("Count")
plt.title("Class Distribution in Dataset")
plt.xticks(rotation=45)
plt.show()


model = Sequential([
    Dense(32, input_dim=X_train.shape[1], activation='relu'),
    BatchNormalization(),
    Dropout(0.3),  # Reduce dropout to prevent underfitting
    Dense(16, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(8, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')  # Output layer
])
model.summary()

option = input("are you ready to start training? (y/n)")

if option.lower() == 'y':
    joblib.dump(scaler, '_scaler2.pkl')

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, mode='max', restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)

    #train model
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping, reduce_lr])

    # eval model
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    print(classification_report(y_test, y_pred_classes, target_names=label_encoder.classes_))

    # Use y_val instead of y_test and index GESTURE_CLASSES with y_pred_classes
    cm = confusion_matrix(y_test, y_pred_classes)  # confusion_matrix

    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='d',
                cmap='Reds',
                xticklabels=label_encoder.classes_,  # Use label_encoder.classes_ for xticklabels
                yticklabels=label_encoder.classes_)  # Use label_encoder.classes_ for yticklabels
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    cmn = (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis])
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(cmn, annot=True, fmt='.0%',
                cmap='Greens',
                xticklabels=label_encoder.classes_,  # Use label_encoder.classes_ for xticklabels
                yticklabels=label_encoder.classes_)  # Use label_encoder.classes_ for yticklabels

    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title("Confusion Matrix")
    plt.show()

    model.save('_model2.keras')
else:
    quit()
