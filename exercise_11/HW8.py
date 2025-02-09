import numpy as np
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, Activation
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import h5py

# 確保 GPU 可用
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # 選擇第一個 GPU
        tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
        # 啟用內存增長模式，避免佔滿整個顯存
        tf.config.experimental.set_memory_growth(gpus[1], True)
        print("Using GPU:", gpus[1])
    except RuntimeError as e:
        print("Error setting GPU:", e)

# 載入 MNIST 資料集
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 資料標準化
x_train = x_train / 255.0
x_test = x_test / 255.0

# 增加通道維度以符合 CNN 的輸入需求
x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]

# 繪製隨機樣本
idx = np.random.randint(0, x_train.shape[0])
plt.imshow(x_train[idx].squeeze(), cmap="gray")
plt.title(f"Label: {y_train[idx]}")
plt.show()

# 構建模型
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
model.summary()

# 分割資料集（80% 訓練，20% 驗證）
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# 編譯模型
model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# 訓練模型
history = model.fit(x_train, y_train, epochs=50, validation_data=(x_val, y_val))
model.save("model.h5")  
print("Model saved successfully as model.h5")
epochs = np.arange(1, len(history.history['loss']) + 1)

# 評估模型
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")


# 繪製訓練損失與驗證損失
plt.plot(epochs, history.history['loss'], 'ko--', label='Training Loss')  # 黑色圓點虛線
plt.xlabel('epoch')  # x軸標籤
plt.ylabel('loss')   # y軸標籤
plt.title("Training Loss Curve")  # 圖標題
plt.xlim((0, 11))
plt.xticks(np.arange(1, 11, step=1))
# plt.grid(True)  # 顯示網格
plt.legend()  # 顯示圖例
plt.show()
