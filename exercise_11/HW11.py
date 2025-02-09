import sys
import numpy as np
import tensorflow as tf
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from PIL import Image

# 載入訓練好的模型
MODEL_PATH = "model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

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
            self.predict_digit(image_path)

    def display_image(self, image_path):
        pixmap = QPixmap(image_path)
        pixmap = pixmap.scaled(200, 200, Qt.KeepAspectRatio)
        self.setPixmap(pixmap)

    def predict_digit(self, image_path):
        img = Image.open(image_path).convert('L').resize((28, 28))  # 轉灰階 & 28x28
        img = np.array(img) / 255.0  # 正規化
        img = img.reshape(1, 28, 28, 1)  # 符合模型輸入
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction)
        self.parent().result_label.setText(f"Result: {predicted_class}")

class MNIST_GUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MNIST Picture Classification GUI")
        self.setGeometry(100, 100, 300, 400)
        
        layout = QVBoxLayout()
        self.drop_area = ImageDropWidget(self)
        self.result_label = QLabel("The Result")
        self.result_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.drop_area)
        layout.addWidget(self.result_label)
        self.setLayout(layout)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MNIST_GUI()
    window.show()
    sys.exit(app.exec_())
