#!/usr/bin/env python3
# 这个文件使用 Python 3.8 运行

import rospy
from grasp.srv import MaskGenerate, MaskGenerateResponse
from grasp.msg import CompressedMask
import socket
import pickle
import array
import cv2
import numpy as np
from cv_bridge import CvBridge

class ROSBridgeServer:
    def __init__(self):
        # 初始化 ROS 节点和服务
        self.service = rospy.Service('generate_mask', MaskGenerate, self.handle_generate_mask)
        self.bridge = CvBridge()
        
        # 创建 socket 服务器用于接收来自 Python 3.11 客户端的数据
        self.socket_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket_server.bind(('localhost', 12347))
        self.socket_server.listen(1)
        print("ROS Bridge Server started, waiting for connection...")

    def list_to_bytes(self, mask_list):
        """将嵌套列表转换为字节序列"""
        # 将二维列表展平为一维
        flat_list = []
        for row in mask_list:
            flat_list.extend(row)
        
        # 创建一个 array.array 对象并转换为字节
        arr = array.array('B')
        for value in flat_list:
            arr.append(1 if value else 0)
        return arr.tobytes()

    def handle_generate_mask(self, req):
        """处理ROS服务请求"""
        try:
            # 将ROS图像消息转换为OpenCV格式
            cv_image = self.bridge.imgmsg_to_cv2(req.image, "bgr8")
            
            # 保存临时图像文件
            temp_image_path = "/tmp/temp_image.jpg"
            cv2.imwrite(temp_image_path, cv_image)
            
            # 连接到 socket client
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect(('localhost', 12348))
            
            # 准备请求数据
            data = pickle.dumps((temp_image_path, req.text_prompt, 
                               req.box_threshold, req.text_threshold))
            
            # 发送数据
            client_socket.send(len(data).to_bytes(4, 'big'))
            client_socket.sendall(data)
            
            # 接收响应
            response_size = int.from_bytes(client_socket.recv(4), 'big')
            response = b''
            while len(response) < response_size:
                packet = client_socket.recv(response_size - len(response))
                response += packet
            
            client_socket.close()
            
            # 解析响应
            masks, boxes, labels, scores = pickle.loads(response)
            print(f"Received {len(masks)} masks")
            
            # 创建 ROS 响应
            ros_response = MaskGenerateResponse()
            
            # 转换每个 mask 为 CompressedMask 消息
            for mask, box, label, score in zip(masks, boxes, labels, scores):
                compressed_mask = CompressedMask()
                compressed_mask.data = self.list_to_bytes(mask)
                compressed_mask.box = [float(x) for x in box]
                compressed_mask.label = label
                compressed_mask.score = float(score)
                compressed_mask.shape = [len(mask), len(mask[0])]
                ros_response.masks.append(compressed_mask)
            
            return ros_response
            
        except Exception as e:
            print(f"Error in handle_generate_mask: {e}")
            return MaskGenerateResponse()

def main():
    rospy.init_node('mask_generate_bridge')
    server = ROSBridgeServer()
    print("Bridge server is ready")
    rospy.spin()

if __name__ == '__main__':
    main() 