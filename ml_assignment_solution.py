
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report, accuracy_score, f1_score

def load_and_preprocess_data():
    data = load_iris()
    X = data.data
    y = tf.keras.utils.to_categorical(data.target, num_classes=3)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return train_test_split(X, y, test_size=0.2, random_state=42)

class CustomActivation(tf.keras.layers.Layer):
    def __init__(self):
        super(CustomActivation, self).__init__()
        self.k0 = tf.Variable(initial_value=0.1, trainable=True, dtype=tf.float32)
        self.k1 = tf.Variable(initial_value=0.1, trainable=True, dtype=tf.float32)

    def call(self, inputs):
        return self.k0 + self.k1 * inputs

def build_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.Dense(10, activation=None),
        CustomActivation(),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    return model

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                        epochs=50, batch_size=16, verbose=1)
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)
    print("Classification Report:")
    print(classification_report(y_true, y_pred))
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    return acc, f1

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    model = build_model(input_shape=(X_train.shape[1],))
    acc, f1 = train_and_evaluate_model(model, X_train, X_test, y_train, y_test)
    print(f"Final Accuracy: {acc}")
    print(f"Final F1-Score: {f1}")
