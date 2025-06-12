import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import cv2
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw

class DigitTestApp:
    def __init__(self, root):
        self.root = root
        self.root.title("手写数字识别系统测试")
        self.root.geometry("800x600")
        self.model = None
        
        # 创建主框架
        self.main_frame = tk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左侧面板：模型加载和标准测试
        self.left_panel = tk.LabelFrame(self.main_frame, text="模型测试", padx=10, pady=10)
        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 加载模型按钮
        self.load_btn = tk.Button(self.left_panel, text="加载模型", command=self.load_model)
        self.load_btn.pack(fill=tk.X, pady=5)
        
        # 模型状态显示
        self.model_status = tk.Label(self.left_panel, text="未加载模型", fg="red")
        self.model_status.pack(pady=5)
        
        # 测试按钮
        self.test_btn = tk.Button(self.left_panel, text="运行标准测试", command=self.run_standard_test, state=tk.DISABLED)
        self.test_btn.pack(fill=tk.X, pady=5)
        
        # 测试结果显示
        self.result_text = tk.Text(self.left_panel, height=10, width=40)
        self.result_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 混淆矩阵按钮
        self.cm_btn = tk.Button(self.left_panel, text="显示混淆矩阵", command=self.show_confusion_matrix, state=tk.DISABLED)
        self.cm_btn.pack(fill=tk.X, pady=5)
        
        # 右侧面板：自定义测试
        self.right_panel = tk.LabelFrame(self.main_frame, text="自定义测试", padx=10, pady=10)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 从文件测试按钮
        self.file_btn = tk.Button(self.right_panel, text="从文件测试", command=self.test_from_file, state=tk.DISABLED)
        self.file_btn.pack(fill=tk.X, pady=5)
        
        # 手写测试按钮
        self.draw_btn = tk.Button(self.right_panel, text="手写测试", command=self.test_by_drawing, state=tk.DISABLED)
        self.draw_btn.pack(fill=tk.X, pady=5)
        
        # 测试图像显示区域
        self.image_frame = tk.Frame(self.right_panel)
        self.image_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.image_label = tk.Label(self.image_frame, text="测试图像将显示在这里")
        self.image_label.pack(fill=tk.BOTH, expand=True)
        
        # 预测结果显示
        self.prediction_label = tk.Label(self.right_panel, text="预测结果: 无", font=("Arial", 14))
        self.prediction_label.pack(pady=10)
        
        # 状态栏
        self.status_bar = tk.Label(root, text="就绪", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # 加载MNIST数据
        self.load_mnist_data()
    
    def load_mnist_data(self):
        try:
            # 加载MNIST测试集
            (_, _), (self.x_test, self.y_test) = mnist.load_data()
            self.x_test_processed = self.preprocess_data(self.x_test)
            self.y_test_cat = to_categorical(self.y_test, 10)
            self.status_bar.config(text="MNIST测试数据加载成功")
        except Exception as e:
            messagebox.showerror("错误", f"无法加载MNIST数据: {str(e)}")
    
    def preprocess_data(self, images):
        # 预处理数据（与训练时一致）
        return images.reshape(-1, 28, 28, 1).astype('float32') / 255
    
    def load_model(self):
        file_path = filedialog.askopenfilename(
            title="选择模型文件",
            filetypes=[("Keras模型", "*.h5"), ("所有文件", "*.*")]
        )
        
        if file_path:
            try:
                self.model = load_model(file_path)
                self.model_status.config(text=f"已加载模型: {os.path.basename(file_path)}", fg="green")
                self.test_btn.config(state=tk.NORMAL)
                self.file_btn.config(state=tk.NORMAL)
                self.draw_btn.config(state=tk.NORMAL)
                self.status_bar.config(text="模型加载成功")
            except Exception as e:
                messagebox.showerror("错误", f"无法加载模型: {str(e)}")
                self.model_status.config(text="模型加载失败", fg="red")
    
    def run_standard_test(self):
        if self.model is None:
            messagebox.showwarning("警告", "请先加载模型")
            return
        
        self.status_bar.config(text="正在运行标准测试...")
        self.root.update()
        
        try:
            # 评估模型
            test_loss, test_accuracy = self.model.evaluate(
                self.x_test_processed, self.y_test_cat, verbose=0)
            
            # 获取所有预测结果
            y_pred = np.argmax(self.model.predict(self.x_test_processed, verbose=0), axis=1)
            
            # 计算分类准确率
            correct = np.sum(y_pred == self.y_test)
            accuracy = correct / len(self.y_test)
            
            # 显示结果
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"标准测试结果:\n\n")
            self.result_text.insert(tk.END, f"样本总数: {len(self.y_test)}\n")
            self.result_text.insert(tk.END, f"正确预测: {correct}\n")
            self.result_text.insert(tk.END, f"准确率: {accuracy:.4f} ({accuracy*100:.2f}%)\n\n")
            self.result_text.insert(tk.END, f"损失值: {test_loss:.4f}\n")
            self.result_text.insert(tk.END, f"评估准确率: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
            
            self.status_bar.config(text="标准测试完成")
            self.cm_btn.config(state=tk.NORMAL)
            
        except Exception as e:
            messagebox.showerror("错误", f"测试过程中出错: {str(e)}")
            self.status_bar.config(text="测试失败")
    
    def show_confusion_matrix(self):
        if self.model is None:
            messagebox.showwarning("警告", "请先加载模型并运行测试")
            return
        
        self.status_bar.config(text="生成混淆矩阵...")
        self.root.update()
        
        try:
            # 获取所有预测结果
            y_pred = np.argmax(self.model.predict(self.x_test_processed, verbose=0), axis=1)
            
            # 计算混淆矩阵
            cm = confusion_matrix(self.y_test, y_pred)
            
            # 创建新窗口显示混淆矩阵
            cm_window = tk.Toplevel(self.root)
            cm_window.title("混淆矩阵")
            cm_window.geometry("600x500")
            
            # 创建图形
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=range(10), yticklabels=range(10))
            plt.xlabel('预测标签')
            plt.ylabel('真实标签')
            plt.title('混淆矩阵')
            
            # 保存临时图像
            temp_file = "confusion_matrix.png"
            plt.savefig(temp_file)
            plt.close()
            
            # 显示图像
            img = Image.open(temp_file)
            img = img.resize((580, 450), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            
            label = tk.Label(cm_window, image=photo)
            label.image = photo  # 保持引用
            label.pack(fill=tk.BOTH, expand=True)
            
            # 关闭窗口时删除临时文件
            cm_window.protocol("WM_DELETE_WINDOW", lambda: (os.remove(temp_file), cm_window.destroy()))
            
            self.status_bar.config(text="混淆矩阵已显示")
            
        except Exception as e:
            messagebox.showerror("错误", f"生成混淆矩阵时出错: {str(e)}")
            self.status_bar.config(text="生成混淆矩阵失败")
    
    def test_from_file(self):
        if self.model is None:
            messagebox.showwarning("警告", "请先加载模型")
            return
        
        file_path = filedialog.askopenfilename(
            title="选择图像文件",
            filetypes=[("图像文件", "*.png;*.jpg;*.jpeg;*.bmp"), ("所有文件", "*.*")]
        )
        
        if file_path:
            try:
                self.status_bar.config(text="正在处理图像...")
                self.root.update()
                
                # 读取图像
                image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                
                if image is None:
                    messagebox.showerror("错误", "无法读取图像文件")
                    self.status_bar.config(text="图像处理失败")
                    return
                
                # 显示原始图像
                img = Image.open(file_path)
                img = img.resize((200, 200), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                
                self.image_label.config(image=photo)
                self.image_label.image = photo  # 保持引用
                
                # 预处理图像
                processed_image = cv2.resize(image, (28, 28))
                processed_image = 255 - processed_image  # 反色（MNIST中数字为白色，背景为黑色）
                processed_image = processed_image.astype('float32') / 255
                processed_image = processed_image.reshape(1, 28, 28, 1)
                
                # 预测
                prediction = self.model.predict(processed_image, verbose=0)
                digit = np.argmax(prediction)
                confidence = np.max(prediction)
                
                # 显示结果
                self.prediction_label.config(text=f"预测结果: {digit}\n置信度: {confidence:.4f}")
                self.status_bar.config(text="图像处理完成")
                
            except Exception as e:
                messagebox.showerror("错误", f"处理图像时出错: {str(e)}")
                self.status_bar.config(text="图像处理失败")
    
    def test_by_drawing(self):
        if self.model is None:
            messagebox.showwarning("警告", "请先加载模型")
            return
        
        # 创建绘图窗口
        draw_window = tk.Toplevel(self.root)
        draw_window.title("手写测试")
        draw_window.geometry("400x500")
        
        # 设置画布
        canvas_width = 280
        canvas_height = 280
        canvas = tk.Canvas(draw_window, width=canvas_width, height=canvas_height, bg="white")
        canvas.pack(pady=10)
        
        # 创建PIL图像用于保存画布内容
        image = Image.new("L", (canvas_width, canvas_height), 0)
        draw = ImageDraw.Draw(image)
        
        # 上一次鼠标位置
        last_x = None
        last_y = None
        
        # 绘制函数
        def paint(event):
            nonlocal last_x, last_y
            x, y = event.x, event.y
            if last_x and last_y:
                canvas.create_line((last_x, last_y, x, y), fill="black", width=15, 
                                  capstyle=tk.ROUND, smooth=True)
                draw.line((last_x, last_y, x, y), fill=255, width=15)
            last_x = x
            last_y = y
        
        def reset(event):
            nonlocal last_x, last_y
            last_x = None
            last_y = None
        
        # 绑定鼠标事件
        canvas.bind("<B1-Motion>", paint)
        canvas.bind("<ButtonRelease-1>", reset)
        
        # 结果标签
        result_label = tk.Label(draw_window, text="预测结果: 无", font=("Arial", 14))
        result_label.pack(pady=10)
        
        # 按钮框架
        button_frame = tk.Frame(draw_window)
        button_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # 识别按钮
        def recognize():
            nonlocal image, result_label
            
            # 处理图像
            img_array = np.array(image)
            img_array = cv2.resize(img_array, (28, 28))
            img_array = 255 - img_array  # 反色
            img_array = img_array.astype('float32') / 255
            img_array = img_array.reshape(1, 28, 28, 1)
            
            # 预测
            prediction = self.model.predict(img_array, verbose=0)
            digit = np.argmax(prediction)
            confidence = np.max(prediction)
            
            # 显示结果
            result_label.config(text=f"预测结果: {digit}\n置信度: {confidence:.4f}")
        
        recognize_btn = tk.Button(button_frame, text="识别", command=recognize)
        recognize_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # 清除按钮
        def clear():
            nonlocal canvas, image, draw
            canvas.delete("all")
            image = Image.new("L", (canvas_width, canvas_height), 0)
            draw = ImageDraw.Draw(image)
            result_label.config(text="预测结果: 无")
        
        clear_btn = tk.Button(button_frame, text="清除", command=clear)
        clear_btn.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=5)

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitTestApp(root)
    root.mainloop()
