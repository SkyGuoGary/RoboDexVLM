#!/usr/bin/env python3
import rospy
import numpy as np
from PIL import Image
import cv2
from grasp.srv import MaskGenerate, GraspPose, set_angle
from sensor_msgs.msg import Image as RosImage
from cv_bridge import CvBridge
import pyrealsense2 as rs
import socket
import pickle
import os
import json
import time
import threading
from scipy.spatial.transform import Rotation as R

class GraspPlanner:
    def __init__(self):
        rospy.init_node('grasp_planner')
        self.bridge = CvBridge()
        
        # 移除相机初始化相关的变量
        self.pipeline = None
        self.align = None
        
        # 存储图像和box信息的类变量
        self.color_image = None
        self.depth_image = None
        self.box_info = None
        # 添加目标位姿存储
        self.target_pose = None
        
        # # fruit on table 
        # home_translation = [-0.5, -0.04, 0.7]
        # rpy_deg = [-176, 1, 85]  # RPY角度值

        # # drawer
        # home_translation = [-0.096, -0.195, 0.7]
        # rpy_deg = [-140, -3, 132]  # RPY角度值

        # box,handle and fruit
        home_translation = [-0.45, 0, 0.7]
        rpy_deg = [-176, 0, 90]  # RPY角度值

        rpy_rad = np.array(rpy_deg) * np.pi / 180.0  # 转换为弧度
        rot_matrix = R.from_euler('xyz', rpy_rad).as_matrix()
        # 将3x3矩阵展平为列表
        rot_list = rot_matrix.flatten().tolist()
        
        self.home_pose = {
            'translation': home_translation,  # 单位：米
            'rotation_matrix': rot_list,
            'score': 1.0  # 设置一个固定的置信度分数
        }
        
        # 添加执行状态变量
        self.execution_completed = False
        self.is_executing = False
        self.wait_thread = None  # 添加等待线程变量
        
        # 等待服务可用
        rospy.loginfo("Waiting for mask generation service...")
        rospy.wait_for_service('generate_mask')
        self.mask_service = rospy.ServiceProxy('generate_mask', MaskGenerate)
        
        rospy.loginfo("Waiting for grasp pose service...")
        rospy.wait_for_service('detect_grasp_pose')
        self.grasp_service = rospy.ServiceProxy('detect_grasp_pose', GraspPose)
        
        # 等待抓手服务可用
        rospy.loginfo("Waiting for hand control service...")
        rospy.wait_for_service('/right_inspire_hand/set_angle')
        self.hand_service = rospy.ServiceProxy('/right_inspire_hand/set_angle', set_angle)
        
        # 初始化socket服务器
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.settimeout(1.0)
        self.server_socket.bind(('10.7.126.92', 12345))
        self.server_socket.listen(1)
        rospy.loginfo("Planner server started, waiting for connection...")

    def capture_images(self):
        """捕获图像并自动处理相机的初始化和关闭"""
        try:
            # 初始化相机
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
            
            # 启动相机
            pipeline.start(config)
            align = rs.align(rs.stream.color)
            
            # 等待相机预热
            rospy.sleep(1.0)
            
            # 捕获图像
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            self.depth_image = np.asanyarray(depth_frame.get_data())
            self.color_image = np.asanyarray(color_frame.get_data())
            
            # 关闭相机
            pipeline.stop()
            
            return self.color_image, self.depth_image
            
        except Exception as e:
            rospy.logerr(f"Error capturing images: {str(e)}")
            # 确保在发生错误时也关闭相机
            if pipeline:
                try:
                    pipeline.stop()
                except:
                    pass
            return None, None

    def generate_mask(self, text_prompt):
        """生成mask并保存box信息"""
        try:
            # 捕获图像
            color_image, _ = self.capture_images()
            if color_image is None:
                return False
            
            # 调用mask生成服务
            rgb_msg = self.bridge.cv2_to_imgmsg(color_image, "bgr8")
            response = self.mask_service(rgb_msg, text_prompt, 0.3, 0.25)
            
            if not response.masks:
                rospy.logwarn("No masks detected")
                return False
            
            # 获取得分最高的mask
            best_mask = max(response.masks, key=lambda x: x.score)
            self.box_info = {
                'box': [int(x) for x in best_mask.box],
                'label': best_mask.label,
                'score': float(best_mask.score)
            }
            
            rospy.loginfo(f"Successfully generated mask and saved box info: {self.box_info}")
            return True
            
        except Exception as e:
            rospy.logerr(f"Error in generate_mask: {str(e)}")
            return False

    def detect_grasp_pose(self):
        """检测抓取位姿"""
        try:
            # 检查是否有必要的数据
            if self.box_info is None:
                rospy.logerr("No box information found. Please run generate_mask first.")
                return False
            
            if self.color_image is None or self.depth_image is None:
                rospy.logerr("No images found. Please run generate_mask first.")
                return False
            
            # 处理图像
            x1, y1, x2, y2 = self.box_info['box']
            mask = np.zeros_like(self.depth_image, dtype=bool)
            mask[y1:y2, x1:x2] = True
            
            depth_masked = self.depth_image.copy()
            depth_masked[~mask] = 2000
            
            rgb_masked = self.color_image.copy()
            rgb_masked[~mask] = 0
            
            # 转换为ROS消息
            rgb_msg = self.bridge.cv2_to_imgmsg(rgb_masked, "bgr8")
            depth_msg = self.bridge.cv2_to_imgmsg(depth_masked)
            
            # 调用抓取服务
            response = self.grasp_service(rgb_msg, depth_msg)
            
            if response.success:
                # 存储目标位姿
                self.target_pose = {
                    'translation': response.translation,
                    'rotation_matrix': response.rotation_matrix,
                    'score': response.score
                }
                # print("grasp pose:", self.target_pose)
                rospy.loginfo("Successfully detected grasp pose")
                return True
            else:
                rospy.logwarn("Failed to get grasp pose")
                return False
                
        except Exception as e:
            rospy.logerr(f"Error in detect_grasp_pose: {str(e)}")
            return False

    def detect_release_pose(self):
        """检测释放位姿"""
        try:
            # 检查是否有必要的数据
            if self.box_info is None:
                rospy.logerr("No box information found. Please run generate_mask first.")
                return False
            
            if self.color_image is None or self.depth_image is None:
                rospy.logerr("No images found. Please run generate_mask first.")
                return False
            
            # 获取box的中心点
            x1, y1, x2, y2 = self.box_info['box']
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # 获取深度信息（单位：毫米）
            depth = self.depth_image[center_y, center_x]
            
            # 使用固定的相机内参
            depth_intrin = rs.intrinsics()
            depth_intrin.width = 640
            depth_intrin.height = 480
            depth_intrin.ppx = 321.285  # 主点x
            depth_intrin.ppy = 237.486  # 主点y
            depth_intrin.fx = 383.970   # 焦距x
            depth_intrin.fy = 383.970   # 焦距y
            depth_intrin.model = rs.distortion.brown_conrady
            depth_intrin.coeffs = [0, 0, 0, 0, 0]  # 畸变系数
            
            # 将像素坐标转换为相机坐标（单位：米）
            camera_point = rs.rs2_deproject_pixel_to_point(depth_intrin, [center_x, center_y], depth)
            
            x = camera_point[0] / 1000.0  # 转换为米
            y = camera_point[1] / 1000.0
            z = camera_point[2] / 1000.0 
            # print("x_old:", x, "y_old:", y, "z_old:", z)
            # 沿中心点与相机中心点连线方向，上升15cm
            # xoverz=x/z
            # yoverz=y/z
            # z_new=z-0.15
            # x_new=xoverz*z_new
            # y_new=yoverz*z_new
            
            x_new=x
            y_new=y
            z_new=z

            # 定义旋转矩阵
            rotation_matrix = [0, 1, 0,
                             0, 0, 1,
                             1, 0, 0]
            
            # 存储目标位姿
            self.target_pose = {
                'translation': tuple([x_new, y_new, z_new]),  # 转换为元组
                'rotation_matrix': rotation_matrix,
                'score': 1.0  # 设置一个固定的置信度分数
            }
            
            # 在成功检测到release pose后添加标记
            self.is_release_pose = True
            
            rospy.loginfo("Successfully detected release pose")
            return True
                
        except Exception as e:
            rospy.logerr(f"Error in detect_release_pose: {str(e)}")
            return False

    def wait_for_execution(self):
        """在单独的线程中等待执行完成"""
        timeout = 60  # 60秒超时
        start_time = time.time()
        while not self.execution_completed and time.time() - start_time < timeout:
            rospy.sleep(0.1)
        
        if not self.execution_completed:
            rospy.logerr("Timeout waiting for execution completion")
        
        self.is_executing = False

    def move_to_target_pose(self):
        """发送目标位姿到tf_convert"""
        try:
            if self.target_pose is None:
                rospy.logerr("No target pose available. Please detect grasp pose first.")
                return False
            
            # 连接到tf_convert并发送数据
            client_socket = None
            try:
                client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client_socket.settimeout(3.0)
                client_socket.connect(('10.7.145.140', 12347))
                
                # 创建命令数据
                command_data = {
                    'pose': self.target_pose
                }
                
                # 如果是release pose，添加特殊标记
                if hasattr(self, 'is_release_pose') and self.is_release_pose:
                    command_data['is_release'] = True
                    command_data['release_height'] = 0.15 # 上升高度（米）
                
                data = pickle.dumps(command_data)
                client_socket.send(len(data).to_bytes(4, 'big'))
                client_socket.sendall(data)
                
                response_size = int.from_bytes(client_socket.recv(4), 'big')
                server_response = b''
                while len(server_response) < response_size:
                    packet = client_socket.recv(response_size - len(server_response))
                    if not packet:
                        raise Exception("Connection closed by server")
                    server_response += packet
                
                # 重置release pose标记
                if hasattr(self, 'is_release_pose'):
                    self.is_release_pose = False
                    rospy.loginfo("Reset release pose flag")
                
                rospy.loginfo("Successfully sent pose to tf_convert")
                return True
                
            except Exception as e:
                rospy.logerr(f"Failed to send pose to tf_convert: {e}")
                return False
            finally:
                if client_socket:
                    client_socket.close()
                
        except Exception as e:
            rospy.logerr(f"Error in move_to_target_pose: {str(e)}")
            return False

    def move_home(self):
        """移动到初始位姿"""
        try:
            # 连接到tf_convert并发送数据
            client_socket = None
            try:
                client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client_socket.settimeout(3.0)
                client_socket.connect(('10.7.145.140', 12347))
                
                # 添加标记表明这是home位姿
                home_command = {
                    'is_home': True,
                    'pose': self.home_pose
                }
                
                data = pickle.dumps(home_command)
                client_socket.send(len(data).to_bytes(4, 'big'))
                client_socket.sendall(data)
                
                response_size = int.from_bytes(client_socket.recv(4), 'big')
                server_response = b''
                while len(server_response) < response_size:
                    packet = client_socket.recv(response_size - len(server_response))
                    if not packet:
                        raise Exception("Connection closed by server")
                    server_response += packet
                
                rospy.loginfo("Successfully sent home pose to tf_convert")
                return True
                
            except Exception as e:
                rospy.logerr(f"Failed to send home pose to tf_convert: {e}")
                return False
            finally:
                if client_socket:
                    client_socket.close()
                
        except Exception as e:
            rospy.logerr(f"Error in move_home: {str(e)}")
            return False

    def control_hand(self, command):
        """控制抓手"""
        try:
            if command == 'grasp':
                # 关闭抓手
                response = self.hand_service(0, 0, 0, 0, 0, 0)
                success = response.angle_accepted
            elif command == 'release':
                # 打开抓手
                response = self.hand_service(1000, 1000, 1000, 1000, 1000, 0)
                success = response.angle_accepted
            
            if success:
                rospy.loginfo(f"Successfully executed hand {command}")
                return True
            else:
                rospy.logerr(f"Failed to execute hand {command}")
                return False
                
        except Exception as e:
            rospy.logerr(f"Error controlling hand: {str(e)}")
            return False

    def run(self):
        """运行服务器循环"""
        while not rospy.is_shutdown():
            try:
                client_socket, address = self.server_socket.accept()
                rospy.loginfo(f"Connected to {address}")
                
                try:
                    # 接收数据
                    data_size = int.from_bytes(client_socket.recv(4), 'big')
                    data = b''
                    while len(data) < data_size:
                        packet = client_socket.recv(data_size - len(data))
                        if not packet:
                            raise Exception("Connection closed by client")
                        data += packet
                    
                    # 解析命令
                    command = pickle.loads(data)
                    success = False
                    
                    if isinstance(command, dict):
                        if command['type'] == 'execution_complete':
                            self.execution_completed = command['status']
                            success = True
                        elif self.is_executing:
                            rospy.logwarn("Cannot execute new command while robot is moving")
                            success = False
                        else:
                            # 处理命令
                            if command['type'] == 'generate_mask':
                                success = self.generate_mask(command['prompt'])
                            elif command['type'] == 'detect_grasp_pose':
                                success = self.detect_grasp_pose()
                            elif command['type'] == 'detect_release_pose':
                                success = self.detect_release_pose()
                            elif command['type'] == 'move_to_target_pose':
                                success = self.move_to_target_pose()
                            elif command['type'] == 'grasp':
                                success = self.control_hand('grasp')
                            elif command['type'] == 'release':
                                success = self.control_hand('release')
                            elif command['type'] == 'move_home':
                                success = self.move_home()
                            elif command['type'] == 'pull':
                                success = self.pull()
                            else:
                                rospy.logwarn(f"Unknown command type: {command['type']}")
                    
                    # 发送响应
                    response = pickle.dumps(success)
                    client_socket.send(len(response).to_bytes(4, 'big'))
                    client_socket.sendall(response)
                    
                except Exception as e:
                    rospy.logerr(f"Error handling client request: {e}")
                finally:
                    client_socket.close()
                    
            except socket.timeout:
                continue
            except Exception as e:
                rospy.logerr(f"Socket error: {e}")
                continue

    def pull(self):
        """执行pull动作"""
        try:
            # 连接到tf_convert并发送数据
            client_socket = None
            try:
                client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client_socket.settimeout(3.0)
                client_socket.connect(('10.7.145.140', 12347))
                
                # 创建pull命令
                pull_command = {
                    'is_pull': True,  # 标记这是一个pull命令
                    'distance': -0.25   # 移动距离（米）
                }
                
                data = pickle.dumps(pull_command)
                client_socket.send(len(data).to_bytes(4, 'big'))
                client_socket.sendall(data)
                
                response_size = int.from_bytes(client_socket.recv(4), 'big')
                server_response = b''
                while len(server_response) < response_size:
                    packet = client_socket.recv(response_size - len(server_response))
                    if not packet:
                        raise Exception("Connection closed by server")
                    server_response += packet
                
                rospy.loginfo("Successfully sent pull command to tf_convert")
                return True
                
            except Exception as e:
                rospy.logerr(f"Failed to send pull command to tf_convert: {e}")
                return False
            finally:
                if client_socket:
                    client_socket.close()
                
        except Exception as e:
            rospy.logerr(f"Error in pull: {str(e)}")
            return False

    def __del__(self):
        """清理资源"""
        try:
            self.server_socket.close()
        except:
            pass

if __name__ == '__main__':
    try:
        planner = GraspPlanner()
        planner.run()
    except rospy.ROSInterruptException:
        pass 