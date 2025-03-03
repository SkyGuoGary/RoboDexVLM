#!/usr/bin/env python
import rospy
import tf2_ros
import geometry_msgs.msg
from geometry_msgs.msg import PoseStamped, TransformStamped
import tf.transformations as tf_trans
import numpy as np
import socket
import pickle
from std_msgs.msg import Bool

class PoseTransformer:
    def __init__(self):
        rospy.init_node('pose_transformer')
        
        # 定义hand到tool0的变换参数
        self.hand_x_offset = -0.02
        self.hand_y_offset = 0.038
        self.hand_z_offset = 0.175
        
        # 更新欧拉角（度数）：roll=0, pitch=30, yaw=0
        self.hand_roll = -45
        self.hand_pitch = 0
        self.hand_yaw = 0.0
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # 移除动态广播器，只使用静态广播器
        self.static_broadcaster = tf2_ros.StaticTransformBroadcaster()
        
        self.pose_pub = rospy.Publisher('/grasp_box_converted_to_base', PoseStamped, queue_size=10)
        self.timer = rospy.Timer(rospy.Duration(0.1), self.broadcast_hand_frame)
        
        # 用于跟踪是否已经发布过grasp pose
        self.grasp_pose_published = False
        self.original_grasp_printed = False
        self.target_pose_printed = False

        # 初始化socket服务器
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 设置socket选项以允许地址重用
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # 设置超时
        self.server_socket.settimeout(30.0)
        self.server_socket.bind(('10.7.145.140', 12347))
        self.server_socket.listen(1)
        rospy.loginfo("TF Convert Server started, waiting for connection...")

    def run(self):
        """运行socket服务器循环"""
        while not rospy.is_shutdown():
            try:
                client_socket, address = self.server_socket.accept()
                rospy.loginfo(f"Connected to {address}")
                
                # 接收数据大小
                data_size = int.from_bytes(client_socket.recv(4), 'big')
                
                # 接收数据
                data = b''
                while len(data) < data_size:
                    packet = client_socket.recv(data_size - len(data))
                    data += packet
                
                # 解析数据
                command = pickle.loads(data)
                
                # 检查是否是向上移动命令
                if isinstance(command, dict) and command.get('is_move_up', False):
                    # 获取当前机器人位姿
                    try:
                        transform = self.tf_buffer.lookup_transform(
                            'base',
                            'tool0_controller',
                            rospy.Time(0),
                            rospy.Duration(1.0)
                        )
                        
                        # 创建新的目标位姿（在当前位置上升0.1m）
                        target_pose = PoseStamped()
                        target_pose.header.frame_id = 'base'
                        target_pose.header.stamp = rospy.Time.now()
                        
                        # 复制当前位置和姿态
                        target_pose.pose.position.x = transform.transform.translation.x
                        target_pose.pose.position.y = transform.transform.translation.y
                        target_pose.pose.position.z = transform.transform.translation.z + 0.1  # 上升0.1m
                        target_pose.pose.orientation = transform.transform.rotation
                        
                        # 发布位姿
                        self.pose_pub.publish(target_pose)
                        
                    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                        rospy.logerr(f"TF查询失败: {str(e)}")
                        
                    # 发送确认消息
                    response = pickle.dumps("success")
                    client_socket.send(len(response).to_bytes(4, 'big'))
                    client_socket.sendall(response)
                elif isinstance(command, dict) and command.get('is_pull', False):
                    try:
                        # 获取当前tool0的位姿
                        current_pose = self.tf_buffer.lookup_transform(
                            'base',
                            'tool0_controller',
                            rospy.Time(0),
                            rospy.Duration(1.0)
                        )
                        
                        # 创建新的目标位姿
                        target_pose = PoseStamped()
                        target_pose.header.frame_id = 'base'
                        target_pose.header.stamp = rospy.Time.now()
                        
                        # 保持当前的z值和方向不变，增加x和y值
                        distance = command.get('distance', 0.15)  # 默认移动0.2米
                        target_pose.pose.position.x = current_pose.transform.translation.x + distance
                        target_pose.pose.position.y = current_pose.transform.translation.y + distance
                        target_pose.pose.position.z = current_pose.transform.translation.z
                        target_pose.pose.orientation = current_pose.transform.rotation
                        
                        # 发布位姿
                        self.pose_pub.publish(target_pose)
                        
                    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                        rospy.logerr(f"TF查询失败: {str(e)}")
                else:
                    # 处理正常的抓取位姿
                    self.pose_callback(command)
                    
                    # 发送确认消息
                    response = pickle.dumps("success")
                    client_socket.send(len(response).to_bytes(4, 'big'))
                    client_socket.sendall(response)
                
                client_socket.close()
                
            except Exception as e:
                rospy.logerr(f"Socket error: {e}")
                continue

    def broadcast_hand_frame(self, event):
        try:
            base_to_tool = self.tf_buffer.lookup_transform(
                'base',
                'tool0_controller',
                rospy.Time(0),
                rospy.Duration(1.0)
            )
            
            hand_transform = TransformStamped()
            hand_transform.header.stamp = rospy.Time.now()
            hand_transform.header.frame_id = 'tool0_controller'
            hand_transform.child_frame_id = 'hand_frame'
            
            # 使用类成员变量设置变换
            hand_transform.transform.translation.x = self.hand_x_offset
            hand_transform.transform.translation.y = self.hand_y_offset
            hand_transform.transform.translation.z = self.hand_z_offset
            
            # 使用新的欧拉角创建四元数
            q = tf_trans.quaternion_from_euler(
                self.hand_roll * np.pi/180.0,
                self.hand_pitch * np.pi/180.0,
                self.hand_yaw * np.pi/180.0,
                'sxyz'
            )
            hand_transform.transform.rotation.x = q[0]
            hand_transform.transform.rotation.y = q[1]
            hand_transform.transform.rotation.z = q[2]
            hand_transform.transform.rotation.w = q[3]
            
            self.static_broadcaster.sendTransform(hand_transform)
            
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(f"广播hand frame失败: {str(e)}")

    def pose_callback(self, data):
        """处理位姿数据"""
        try:
            # 检查是否是home位姿
            if isinstance(data, dict) and data.get('is_home', False):
                # 直接创建PoseStamped消息
                home_pose = data['pose']
                target_pose = PoseStamped()
                target_pose.header.frame_id = 'base'
                target_pose.header.stamp = rospy.Time.now()
                
                # 从旋转矩阵创建四元数
                rot_matrix = np.array(home_pose['rotation_matrix']).reshape(3, 3)
                quat = tf_trans.quaternion_from_matrix(np.vstack([
                    np.hstack([rot_matrix, np.zeros((3, 1))]),
                    [0, 0, 0, 1]
                ]))
                
                # 设置位置和方向
                target_pose.pose.position.x = home_pose['translation'][0]
                target_pose.pose.position.y = home_pose['translation'][1]
                target_pose.pose.position.z = home_pose['translation'][2]
                target_pose.pose.orientation.x = quat[0]
                target_pose.pose.orientation.y = quat[1]
                target_pose.pose.orientation.z = quat[2]
                target_pose.pose.orientation.w = quat[3]
                
                # 发布位姿
                self.pose_pub.publish(target_pose)
                return
            
            # 提取pose和命令信息
            if isinstance(data, dict):
                camera_pose = data['pose']
                is_release = data.get('is_release', False)
                release_height = data.get('release_height', 0.2)  # 默认上升0.2米
            else:
                camera_pose = data
                is_release = False
                release_height = 0.2
            
            # 1. 首先获取camera到base的转换
            camera_to_base = self.tf_buffer.lookup_transform(
                'base',
                'camera_color_optical_frame',
                rospy.Time(0),
                rospy.Duration(1.0)
            )
            
            # 2. 创建相机坐标系下抓取位姿的变换矩阵
            grasp_camera_matrix = np.eye(4)
            grasp_camera_matrix[0:3, 0:3] = np.array(camera_pose['rotation_matrix']).reshape(3, 3)
            grasp_camera_matrix[0:3, 3] = camera_pose['translation']
            
            # 创建camera到base的变换矩阵
            camera_to_base_matrix = tf_trans.quaternion_matrix([
                camera_to_base.transform.rotation.x,
                camera_to_base.transform.rotation.y,
                camera_to_base.transform.rotation.z,
                camera_to_base.transform.rotation.w
            ])
            camera_to_base_matrix[0:3,3] = [
                camera_to_base.transform.translation.x,
                camera_to_base.transform.translation.y,
                camera_to_base.transform.translation.z
            ]
            
            # 将抓取位姿转换到base坐标系
            grasp_base_matrix = np.dot(camera_to_base_matrix, grasp_camera_matrix)
            
            # 只在第一次打印原始grasp box位姿
            if not self.original_grasp_printed:
                grasp_pos = tf_trans.translation_from_matrix(grasp_base_matrix)
                grasp_quat = tf_trans.quaternion_from_matrix(grasp_base_matrix)
                grasp_rpy = tf_trans.euler_from_quaternion(grasp_quat)
                
                print("\n========== 原始抓取位姿（相对于base） ==========")
                print(f"位置 (x,y,z): {grasp_pos}")
                print(f"四元数 (x,y,z,w): {grasp_quat}")
                print(f"欧拉角 (r,p,y) [度]: {[angle/np.pi*180 for angle in grasp_rpy]}")
                print("============================================\n")
                
                self.original_grasp_printed = True
            
            # 创建坐标轴变换矩阵
            # 这里是因为tool0的坐标轴和grasp_box的坐标轴不一致
            axis_change = np.array([
                [0, 0, 1, 0],  # 新的x轴 = 原来的z轴
                [0, -1, 0, 0], # 新的y轴 = 原来的-y轴
                [1, 0, 0, 0],  # 新的z轴 = 原来的x轴
                [0, 0, 0, 1]
            ])
            
            # 应用坐标轴变换
            grasp_base_matrix = np.dot(grasp_base_matrix, axis_change)
            
            # 创建tool0到hand的变换矩阵
            tool_to_hand = np.eye(4)
            q_hand = tf_trans.quaternion_from_euler(
                self.hand_roll * np.pi/180.0,
                self.hand_pitch * np.pi/180.0,
                self.hand_yaw * np.pi/180.0,
                'sxyz'
            )
            rot_matrix = tf_trans.quaternion_matrix(q_hand)
            tool_to_hand[:3,:3] = rot_matrix[:3,:3]
            tool_to_hand[0,3] = self.hand_x_offset
            tool_to_hand[1,3] = self.hand_y_offset
            tool_to_hand[2,3] = self.hand_z_offset
            
            # 计算hand到tool0的变换（求逆）
            hand_to_tool = np.linalg.inv(tool_to_hand)
            
            # 计算tool0的目标位置（在base坐标系下）
            target_matrix = np.dot(grasp_base_matrix, hand_to_tool)
            
            # 6. 从矩阵提取位置和姿态
            target_pos = tf_trans.translation_from_matrix(target_matrix)
            target_quat = tf_trans.quaternion_from_matrix(target_matrix)
            
            # 创建最终的目标位姿
            target_pose = PoseStamped()
            target_pose.header.frame_id = 'base'
            target_pose.header.stamp = rospy.Time.now()
            
            # 设置基本位姿
            target_pose.pose.position.x = target_pos[0]
            target_pose.pose.position.y = target_pos[1]
            target_pose.pose.position.z = target_pos[2]
            target_pose.pose.orientation.x = target_quat[0]
            target_pose.pose.orientation.y = target_quat[1]
            target_pose.pose.orientation.z = target_quat[2]
            target_pose.pose.orientation.w = target_quat[3]
            
            # 如果是release pose，调整高度
            if is_release:
                target_pose.pose.position.z += release_height
                rospy.loginfo(f"Release pose detected - Adding {release_height}m in Z direction")
                rospy.loginfo(f"Original Z: {target_pos[2]}, New Z: {target_pose.pose.position.z}")
            
            # 只在第一次打印目标位姿
            if not self.target_pose_printed:
                print("\n========== Tool0目标位姿（相对于base） ==========")
                print(f"位置 (x,y,z): {target_pos}")
                target_rpy = tf_trans.euler_from_quaternion(target_quat)
                print(f"欧拉角 (r,p,y) [度]: {[r/np.pi*180 for r in target_rpy]}")
                print("============================================\n")
                
                self.target_pose_printed = True
            
            self.pose_pub.publish(target_pose)
            
            # 只在第一次收到消息时发布静态TF
            if not self.grasp_pose_published:
                grasp_transform = TransformStamped()
                # 对于静态TF，使用时间0
                grasp_transform.header.stamp = rospy.Time(0)
                grasp_transform.header.frame_id = "base"
                grasp_transform.child_frame_id = "grasp_pose"
                
                grasp_pos = tf_trans.translation_from_matrix(grasp_base_matrix)
                grasp_quat = tf_trans.quaternion_from_matrix(grasp_base_matrix)
                
                grasp_transform.transform.translation.x = grasp_pos[0]
                grasp_transform.transform.translation.y = grasp_pos[1]
                grasp_transform.transform.translation.z = grasp_pos[2]
                grasp_transform.transform.rotation.x = grasp_quat[0]
                grasp_transform.transform.rotation.y = grasp_quat[1]
                grasp_transform.transform.rotation.z = grasp_quat[2]
                grasp_transform.transform.rotation.w = grasp_quat[3]
                
                # 为确保TF被正确发布，可以多发送几次
                for _ in range(3):
                    self.static_broadcaster.sendTransform(grasp_transform)
                    rospy.sleep(0.1)  # 短暂延时
                
                self.grasp_pose_published = True
                rospy.loginfo("已发布静态grasp_pose TF")
                
                # 打印位置信息以便确认
                print("Grasp pose position:", grasp_pos)
                print("Grasp pose orientation (quaternion):", grasp_quat)
            
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(f"TF转换失败: {str(e)}")

    def transform_pose(self, pose, transform):
        # 从四元数创建旋转矩阵
        pose_rotation = tf_trans.quaternion_matrix([
            pose.pose.orientation.x,
            pose.pose.orientation.y,
            pose.pose.orientation.z,
            pose.pose.orientation.w
        ])
        
        # 设置平移部分
        pose_rotation[0:3, 3] = [
            pose.pose.position.x,
            pose.pose.position.y,
            pose.pose.position.z
        ]
        
        # 从四元数创建变换矩阵
        transform_rotation = tf_trans.quaternion_matrix([
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
            transform.transform.rotation.w
        ])
        
        # 设置平移部分
        transform_rotation[0:3, 3] = [
            transform.transform.translation.x,
            transform.transform.translation.y,
            transform.transform.translation.z
        ]
        
        # 进行矩阵乘法得到转换后的位姿
        result_mat = np.dot(transform_rotation, pose_rotation)
        
        # 从矩阵中提取位置和方向
        translation = tf_trans.translation_from_matrix(result_mat)
        quaternion = tf_trans.quaternion_from_matrix(result_mat)
        
        # 创建转换后的位姿消息
        transformed_pose = PoseStamped()
        transformed_pose.header = pose.header
        transformed_pose.pose.position.x = translation[0]
        transformed_pose.pose.position.y = translation[1]
        transformed_pose.pose.position.z = translation[2]
        transformed_pose.pose.orientation.x = quaternion[0]
        transformed_pose.pose.orientation.y = quaternion[1]
        transformed_pose.pose.orientation.z = quaternion[2]
        transformed_pose.pose.orientation.w = quaternion[3]
        
        return transformed_pose

if __name__ == '__main__':
    try:
        transformer = PoseTransformer()
        transformer.run()
    except rospy.ROSInterruptException:
        pass
