import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageDraw, ImageTk
import cv2

class DigitRecognizer:
    def __init__(self):
        # 加载MNIST数据集
        (self.X_train, self.y_train), (self.X_test, self.y_test) = mnist.load_data()
        
        # 数据预处理
        self.X_train = self.X_train.reshape(self.X_train.shape[0], 28, 28, 1).astype('float32') / 255
        self.X_test = self.X_test.reshape(self.X_test.shape[0], 28, 28, 1).astype('float32') / 255
        self.y_train = to_categorical(self.y_train)
        self.y_test = to_categorical(self.y_test)
        
        # 初始化模型
        self.model = None
        
    def build_model(self):
        # 构建CNN模型
        self.model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(10, activation='softmax')
        ])
        
        # 编译模型
        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
        
        return self.model
    
    def train_model(self, epochs=5, batch_size=64):
        if self.model is None:
            self.build_model()
        
        # 训练模型
        history = self.model.fit(
            self.X_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(self.X_test, self.y_test)
        )
        
        # 评估模型
        test_loss, test_acc = self.model.evaluate(self.X_test, self.y_test)
        print(f'测试准确率: {test_acc:.4f}')
        
        return history
    
    def save_model(self, filename='digit_model.h5'):
        if self.model is not None:
            self.model.save(filename)
            print(f'模型已保存到 {filename}')
    
    def load_model(self, filename='digit_model.h5'):
        try:
            self.model = load_model(filename)
            print(f'已加载模型 {filename}')
            return True
        except:
            print('无法加载模型，请确保模型文件存在')
            return False
    
    def predict_digit(self, image):
        if self.model is None:
            print('请先训练或加载模型')
            return None
        
        # 预处理图像
        image = cv2.resize(image, (28, 28))
        image = image.reshape(1, 28, 28, 1).astype('float32') / 255
        
        # 预测
        prediction = self.model.predict(image)
        digit = np.argmax(prediction)
        confidence = np.max(prediction)
        
        return digit, confidence
    
    def predict_from_file(self, file_path):
        # 从文件加载图像
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print('无法加载图像')
            return None
        
        return self.predict_digit(image)

class DrawingApp:
    def __init__(self, root, recognizer):
        self.root = root
        self.recognizer = recognizer
        self.root.title("手写数字识别")
        
        # 设置画布
        self.canvas_width = 200
        self.canvas_height = 200
        self.canvas = tk.Canvas(root, width=self.canvas_width, height=self.canvas_height, bg="white")
        self.canvas.pack(side=tk.LEFT, padx=10, pady=10)
        
        # 创建PIL图像用于保存画布内容
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), 0)
        self.draw = ImageDraw.Draw(self.image)
        
        # 绑定鼠标事件
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)
        
        # 按钮框架
        button_frame = tk.Frame(root)
        button_frame.pack(side=tk.RIGHT, padx=10, pady=10)
        
        # 识别按钮
        self.recognize_button = tk.Button(button_frame, text="识别", command=self.recognize)
        self.recognize_button.pack(fill=tk.X, pady=5)
        
        # 清除按钮
        self.clear_button = tk.Button(button_frame, text="清除", command=self.clear)
        self.clear_button.pack(fill=tk.X, pady=5)
        
        # 结果标签
        self.result_label = tk.Label(button_frame, text="结果: ", font=("Arial", 16))
        self.result_label.pack(pady=20)
        
        # 加载/训练模型按钮
        model_frame = tk.Frame(button_frame)
        model_frame.pack(fill=tk.X, pady=10)
        
        self.load_button = tk.Button(model_frame, text="加载模型", command=self.load_model)
        self.load_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        
        self.train_button = tk.Button(model_frame, text="训练模型", command=self.train_model)
        self.train_button.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=2)
        
        # 上一次鼠标位置
        self.last_x = None
        self.last_y = None
    
    def paint(self, event):
        # 在画布上绘制
        x, y = event.x, event.y
        if self.last_x and self.last_y:
            self.canvas.create_line((self.last_x, self.last_y, x, y), fill="black", width=10, 
                                    capstyle=tk.ROUND, smooth=True)
            self.draw.line((self.last_x, self.last_y, x, y), fill=255, width=10)
        self.last_x = x
        self.last_y = y
    
    def reset(self, event):
        self.last_x = None
        self.last_y = None
    
    def clear(self):
        # 清除画布
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="结果: ")
    
    def recognize(self):
        # 识别画布上的数字
        if self.recognizer.model is None:
            self.result_label.config(text="结果: 请先加载或训练模型")
            return
        
        # 处理图像
        img_array = np.array(self.image)
        img_array = cv2.resize(img_array, (28, 28))
        
        # 反色（因为MNIST中的数字是白色背景黑色数字）
        img_array = 255 - img_array
        
        # 预测
        digit, confidence = self.recognizer.predict_digit(img_array)
        self.result_label.config(text=f"结果: {digit}\n置信度: {confidence:.4f}")
    
    def load_model(self):
        file_path = filedialog.askopenfilename(filetypes=[("Model files", "*.h5")])
        if file_path:
            success = self.recognizer.load_model(file_path)
            if success:
                self.result_label.config(text="结果: 模型已加载")
    
    def train_model(self):
        # 创建训练窗口
        train_window = tk.Toplevel(self.root)
        train_window.title("训练模型")
        train_window.geometry("300x150")
        
        # 训练参数
        tk.Label(train_window, text="训练轮数:").grid(row=0, column=0, padx=10, pady=10)
        epochs_entry = tk.Entry(train_window)
        epochs_entry.insert(0, "5")
        epochs_entry.grid(row=0, column=1, padx=10, pady=10)
        
        tk.Label(train_window, text="批次大小:").grid(row=1, column=0, padx=10, pady=10)
        batch_entry = tk.Entry(train_window)
        batch_entry.insert(0, "64")
        batch_entry.grid(row=1, column=1, padx=10, pady=10)
        
        def start_training():
            epochs = int(epochs_entry.get())
            batch_size = int(batch_entry.get())
            
            # 显示训练进度窗口
            progress_window = tk.Toplevel(self.root)
            progress_window.title("训练进度")
            progress_window.geometry("300x100")
            
            progress_label = tk.Label(progress_window, text="正在训练模型...")
            progress_label.pack(pady=20)
            
            # 训练模型
            self.root.update()
            history = self.recognizer.train_model(epochs=epochs, batch_size=batch_size)
            
            # 保存模型
            self.recognizer.save_model()
            
            progress_label.config(text="训练完成!")
            self.result_label.config(text="结果: 模型已训练并保存")
            
            # 关闭窗口
            progress_window.after(1000, progress_window.destroy)
            train_window.destroy()
        
        train_button = tk.Button(train_window, text="开始训练", command=start_training)
        train_button.grid(row=2, column=0, columnspan=2, pady=10)

def main():
    # 创建识别器实例
    recognizer = DigitRecognizer()
    
    # 尝试加载预训练模型
    recognizer.load_model()
    
    # 创建GUI应用
    root = tk.Tk()
    app = DrawingApp(root, recognizer)
    root.geometry("400x250")
    root.mainloop()

if __name__ == "__main__":
    main()
