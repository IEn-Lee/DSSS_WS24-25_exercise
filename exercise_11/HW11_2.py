import sys
import os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization, Activation
from sklearn.model_selection import train_test_split
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap

from PyQt5.QtCore import Qt
from PIL import Image

# 是否進行模型訓練
TRAIN_MODEL = True  # 設為 True 會重新訓練並轉換 TFLite
epoch = 15

# 設定 TensorFlow GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 如果你想使用特定的GPU，取消注释并设置正确的GPU ID
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("Using GPU:", gpus[0])
    except RuntimeError as e:
        print("Error setting GPU:", e)

# 載入 Fashion MNIST 數據集
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # 正規化

# 增加通道維度
x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]

# 分割訓練/驗證數據
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# 建立 CNN 模型
def build_model():
    model = Sequential()

    # Stage 1: Conv3x3 & ReLU & MaxPooling
    model.add(Conv2D(8, (3, 3), input_shape=(28, 28, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    # Stage 2: Conv3x3 & ReLU & MaxPooling
    model.add(Conv2D(16, (3, 3), padding="same"))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    # Stage 3: Conv3x3 & ReLU
    model.add(Conv2D(32, (3, 3), padding="same"))
    model.add(Activation('relu'))

    # Stage 4: Flatten
    model.add(Flatten())

    # Stage 5: Dense & ReLU & Dropout
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    # Stage 6: Dense & Softmax
    model.add(Dense(10))
    model.add(Activation('softmax'))

    return model

model = build_model()

TFLITE_MODEL_PATH = "fashion_mnist_model.tflite"

if TRAIN_MODEL:
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("TensorFlow is using GPU:", tf.config.list_physical_devices('GPU')) # 更正：使用 tf.config.list_physical_devices

    print("Training model...")
    model = build_model()
    model.summary()
    model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(x_train, y_train, epochs=epoch, validation_data=(x_val, y_val))

    # 顯示訓練/驗證損失與準確率
    epochs = range(1, epoch + 1) # 修改：epochs = range(1, epoch + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.history['loss'], 'r-', label='Training Loss')
    plt.plot(epochs, history.history['val_loss'], 'b--', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training & Validation Loss')

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.history['accuracy'], 'g-', label='Training Accuracy')
    plt.plot(epochs, history.history['val_accuracy'], 'm--', label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training & Validation Accuracy')
    plt.show() # 添加 plt.show() 以显示图像

    # 轉換為 TensorFlow Lite 模型
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(TFLITE_MODEL_PATH, "wb") as f:
        f.write(tflite_model)
    print("Model training complete and saved as TFLite.")

# 載入 TFLite 模型
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 定義 Fashion MNIST 類別名稱
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

class ImageDropWidget(QLabel):
    def __init__(self, parent):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setText("Drop an image here")
        self.setStyleSheet("border: 2px dashed gray")
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            image_path = urls[0].toLocalFile()
            self.display_image(image_path)
            self.predict_image(image_path)

    def display_image(self, image_path):
        pixmap = QPixmap(image_path)
        pixmap = pixmap.scaled(200, 200, Qt.KeepAspectRatio)
        self.setPixmap(pixmap)

    def predict_image(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (28, 28))
        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=(0, -1))

        interpreter.set_tensor(input_details[0]['index'], image)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        prediction = np.argmax(output_data)

        predicted_label = CLASS_NAMES[prediction]  # 轉換為類別名稱
        self.parent().result_label.setText(f"Prediction: {predicted_label}")


class FashionMNIST_GUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fashion MNIST Classifier")
        self.setGeometry(100, 100, 300, 400)

        layout = QVBoxLayout()
        self.drop_area = ImageDropWidget(self)
        self.result_label = QLabel("Prediction: ?")
        self.result_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.drop_area)
        layout.addWidget(self.result_label)
        self.setLayout(layout)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FashionMNIST_GUI()
    window.show()
    sys.exit(app.exec_())
