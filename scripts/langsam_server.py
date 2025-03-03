# 这个文件使用 Python 3.11 运行
import os
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['SSL_CERT_VERIFY'] = '0'

# 在导入其他库之前添加以下代码
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

import socket
import pickle
import numpy as np
from PIL import Image
from lang_sam import LangSAM
import cv2
import matplotlib
matplotlib.use('TkAgg')  # 改用 TkAgg 后端
import matplotlib.pyplot as plt

class MaskGenerationClient:
    def __init__(self):
        # 初始化模型
        print("Initializing LangSAM model...")
        self.model = LangSAM(sam_type="sam2.1_hiera_small")
        print("LangSAM model initialized.")
        
        # 初始化 socket 服务器
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(('localhost', 12348))
        self.server_socket.listen(1)
        print("Mask Generation Client started, waiting for connection...")

    def convert_to_python_types(self, data):
        """将numpy数组转换为Python原生类型"""
        if isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, (list, tuple)):
            return [self.convert_to_python_types(item) for item in data]
        elif isinstance(data, dict):
            return {key: self.convert_to_python_types(value) for key, value in data.items()}
        return data

    def generate_masks(self, image_path, text_prompt, box_threshold, text_threshold):
        """生成 mask"""
        try:
            # 加载并处理图像
            image = Image.open(image_path).convert("RGB")
            # image = self.resize_image_height_700(image)
            
            # 生成 mask
            results = self.model.predict(
                images_pil=[image],
                texts_prompt=[text_prompt],
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )
            results = results[0]
            
            if not len(results["masks"]):
                print("No masks detected.")
                return [], [], [], []
            
            # 在返回结果之前添加可视化代码
            self.visualize_masks(results["masks"])
            
            # 转换numpy数组为Python列表
            masks = self.convert_to_python_types(results["masks"])
            boxes = self.convert_to_python_types(results["boxes"])
            labels = results["labels"]
            scores = self.convert_to_python_types(results["scores"])
            
            print(f"Generated {len(masks)} masks")
            return masks, boxes, labels, scores
            
        except Exception as e:
            print(f"Error in generate_masks: {e}")
            return [], [], [], []

    def visualize_masks(self, masks):
        """保存检测到的masks到文件"""
        n_masks = len(masks)
        if n_masks == 0:
            return
        
        # 计算显示网格的行列数
        n_cols = min(3, n_masks)
        n_rows = (n_masks + n_cols - 1) // n_cols
        
        # 创建一个大的画布
        cell_size = 200  # 每个mask的显示大小
        canvas = np.zeros((cell_size * n_rows, cell_size * n_cols), dtype=np.uint8)
        
        # 填充每个mask
        for i, mask in enumerate(masks):
            row = i // n_cols
            col = i % n_cols
            
            # 将mask转换为uint8类型并缩放到0-255
            mask_img = (mask * 255).astype(np.uint8)
            
            # 调整大小以适应画布单元格
            mask_resized = cv2.resize(mask_img, (cell_size, cell_size))
            
            # 将调整后的mask放入画布
            y_start = row * cell_size
            x_start = col * cell_size
            canvas[y_start:y_start+cell_size, x_start:x_start+cell_size] = mask_resized
        
        # 保存结果到文件
        save_path = '../assets/detected_masks.png'
        cv2.imwrite(save_path, canvas)
        print(f"Masks saved to {save_path}")

    def resize_image_height_700(self, image):
        width, height = image.size
        if height <= 700:
            return image
        scale = 700.0 / height
        new_width = int(width * scale)
        return image.resize((new_width, 700), Image.LANCZOS)

    def start(self):
        try:
            while True:
                try:
                    client_socket, address = self.server_socket.accept()
                    print(f"Connected to {address}")
                    
                    try:
                        # 接收数据
                        data_size = int.from_bytes(client_socket.recv(4), 'big')
                        data = b''
                        while len(data) < data_size:
                            packet = client_socket.recv(data_size - len(data))
                            data += packet
                        
                        # 处理请求
                        image_path, text_prompt, box_threshold, text_threshold = pickle.loads(data)
                        
                        # 生成 mask
                        masks, boxes, labels, scores = self.generate_masks(
                            image_path, text_prompt, box_threshold, text_threshold)
                        
                        # 发送响应
                        response = pickle.dumps((masks, boxes, labels, scores))
                        client_socket.send(len(response).to_bytes(4, 'big'))
                        client_socket.sendall(response)
                        
                    finally:
                        client_socket.close()
                    
                except Exception as e:
                    print(f"Error in connection handling: {e}")
                    continue
        
        except KeyboardInterrupt:
            print("\nShutting down server...")
        finally:
            self.server_socket.close()

    def __del__(self):
        if hasattr(self, 'server_socket'):
            self.server_socket.close()

if __name__ == '__main__':
    client = MaskGenerationClient()
    client.start() 