import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix

from keras._tf_keras.keras.models import Sequential, load_model
from keras._tf_keras.keras.layers import Dense, Dropout, BatchNormalization
from keras._tf_keras.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras._tf_keras.keras.optimizers import Adam


class GestureModel: 
    def __init__(self):
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.scaler = MinMaxScaler()
        self.class_counts = None
        self.num_classes = None
        self.label_encoder = None  # 

    def load_data(self, filename="closer_data_coords.csv"): 
        ##load data
        df = pd.read_csv(os.path.join(os.getcwd(), filename))

        # drop cols, z-coords, and nans
        df.drop(columns=['source_file', 'source_directory'], inplace=True, errors='ignore')
        df.columns = [col.replace('.', '_') for col in df.columns]
        df = df.drop(columns=[c for c in df.columns if '_z' in c], errors="ignore")
        df.dropna(inplace=True)

        # encode labels
        self.label_encoder = LabelEncoder()
        df['gesture_class'] = self.label_encoder.fit_transform(df['gesture_class'])
        self.num_classes = len(self.label_encoder.classes_)

        # features / labels
        X = df.drop(columns=['gesture_class']).apply(pd.to_numeric, errors='coerce')
        y = df['gesture_class']

        X_train, X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.40, random_state=42, stratify=y
        )

        ## fit params
        self.X_train = self.scaler.fit_transform(X_train)
        self.X_test = self.scaler.transform(X_test)
        self.class_counts = df['gesture_class'].value_counts()

    def display_class_distribution(self):
        if self.class_counts is not None: 
            print("Class Distribution:\n", self.class_counts)
            plt.figure(figsize=(8, 5))
            self.class_counts.plot(kind='bar', color='c')
            plt.xlabel("Gesture Class")
            plt.ylabel("Count")
            plt.title("Class Distribution")
            plt.xticks(rotation=45)
            plt.show()

    def update_gesture_model(self):
        self.model = Sequential([
            Dense(32, input_dim=self.X_train.shape[1], activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(16, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(8, activation='relu'),
            Dropout(0.3),
            Dense(self.num_classes, activation='softmax')
        ])

    def print_gesture_model_summary(self):
        if self.model:
            self.model.summary()

    def train_model(self):
        
        joblib.dump(self.scaler, 'gesture_model_scaler.pkl')

        self.model.compile(optimizer=Adam(learning_rate=0.001),
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

        early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, mode='max', restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)

        history = self.model.fit(
            self.X_train, self.y_train,
            epochs=100, batch_size=32,
            validation_data=(self.X_test, self.y_test),
            callbacks=[early_stopping, reduce_lr]
        )

        # eval
        self.y_pred = np.argmax(self.model.predict(self.X_test), axis=1)

        self.model.save('gesture_model.keras')

    def retrain_model(self, new_data_file):
        """Load new data, retrain from saved model, update scaler + weights."""
        # Load previous model
        self.model = load_model("gesture_model.keras")
        self.scaler = joblib.load("gesture_model_scaler.pkl")

        # Load new dataset
        self.load_data(new_data_file)

        # Re-fit
        self.model.fit(self.X_train, self.y_train, epochs=20, batch_size=32, validation_data=(self.X_test, self.y_test))
        self.model.save("gesture_model.keras")

    def print_model_accuracy(self):
        print(classification_report(self.y_test, self.y_pred, target_names=self.label_encoder.classes_))

        cm = confusion_matrix(self.y_test, self.y_pred)

        # confusion matrix
        plt.figure(figsize=(8, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                    xticklabels=self.label_encoder.classes_,
                    yticklabels=self.label_encoder.classes_)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()

        # normalize comfusion matrix
        cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(8, 8))
        sns.heatmap(cmn, annot=True, fmt='.0%', cmap='Greens',
                    xticklabels=self.label_encoder.classes_,
                    yticklabels=self.label_encoder.classes_)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Normalized Confusion Matrix")
        plt.show()

