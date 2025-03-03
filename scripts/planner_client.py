#!/usr/bin/env python3
# 导入 GPT4Requester 类
from request_gpt4 import GPT4Requester

# 其他导入
import socket
import pickle
import time
import rospy
import pyrealsense2 as rs
import numpy as np
from cv_bridge import CvBridge

class PlannerClient:
    def __init__(self):
        self.server_address = ('10.7.126.92', 12345)
        self.bridge = CvBridge()
        
        # 添加socket服务器来接收执行完成信号
        self.execution_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.execution_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.execution_server.settimeout(1.0)
        # 重开机需要修改IP地址
        self.execution_server.bind(('10.7.126.92', 12346))  # 使用不同的端口
        self.execution_server.listen(1)
        print("Execution server started on port 12346")
    
    def capture_images(self):
        """捕获图像并自动处理相机的初始化和关闭"""
        pipeline = None
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
            
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            return color_image, depth_image
            
        except Exception as e:
            print(f"Error capturing images: {str(e)}")
            return None, None
        
        finally:
            # 确保在任何情况下都关闭相机
            if pipeline:
                try:
                    pipeline.stop()
                except:
                    pass

    def wait_for_execution(self, timeout=60):
        """等待执行完成信号"""
        print("Waiting for execution complete signal...")
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                client_socket, address = self.execution_server.accept()
                try:
                    data_size = int.from_bytes(client_socket.recv(4), 'big')
                    data = b''
                    while len(data) < data_size:
                        packet = client_socket.recv(data_size - len(data))
                        if not packet:
                            break
                        data += packet
                    
                    command = pickle.loads(data)
                    if isinstance(command, dict) and command['type'] == 'execution_complete':
                        print("Received execution complete signal")
                        return True
                finally:
                    client_socket.close()
            except socket.timeout:
                continue
            except Exception as e:
                print(f"Error waiting for execution: {e}")
        
        print("Timeout waiting for execution complete signal")
        return False
    
    def send_command(self, command):
        """发送命令到planner服务器"""
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            client_socket.connect(self.server_address)
            
            # 发送命令
            data = pickle.dumps(command)
            client_socket.send(len(data).to_bytes(4, 'big'))
            client_socket.sendall(data)
            
            # 接收响应
            response_size = int.from_bytes(client_socket.recv(4), 'big')
            response = b''
            while len(response) < response_size:
                packet = client_socket.recv(response_size - len(response))
                if not packet:
                    raise Exception("Connection closed by server")
                response += packet
            
            return pickle.loads(response)
            
        finally:
            client_socket.close()
    
    def execute_task(self, task_name, **kwargs):
        """执行单个任务"""
        command = {'type': task_name}
        command.update(kwargs)
        
        print(f"\nExecuting task: {task_name}")
        
        # 在generate_mask之前自动执行move_home
        if task_name == 'generate_mask':
            print("Executing automatic move_home before mask generation")
            if not self.execute_task('move_home'):
                print("Failed to complete automatic move_home")
                return False
        
        success = self.send_command(command)
        
        if success:
            print(f"Task {task_name} initiated successfully")
            # 如果是移动任务，等待执行完成
            if task_name in ['move_to_target_pose', 'move_home', 'pull']:
                if not self.wait_for_execution():
                    print(f"Failed to complete {task_name}: execution timeout")
                    return False
            
            print(f"Successfully completed {task_name}")
        else:
            print(f"Failed to initiate {task_name}")
            return False
        
        return True
    
    def execute_pipeline(self, tasks):
        """执行一系列任务"""

        for task in tasks:
            task_name = task.get('name')
            task_params = task.get('params', {})
            
            if not self.execute_task(task_name, **task_params):
                print(f"Pipeline failed at task: {task_name}")
                return False
            
            # 在任务之间添加短暂延时
            time.sleep(task.get('delay', 1))
        
        print("\nPipeline completed successfully")
        return True
    
    def __del__(self):
        """清理资源"""
        try:
            self.execution_server.close()
        except:
            pass

    def parse_skill_sequence(self, response):
        """解析GPT-4返回的技能序列，转换为任务流程"""
        # 分割技能序列
        skills = response.split('##')[1:-1]  # 去掉首尾空字符串
        tasks = []
        
        for skill in skills:
            if not skill:
                continue
                
            # 解析技能编号和参数
            if '(' in skill:
                skill_num = skill[:skill.find('(')].strip()
                param = skill[skill.find('(')+1:skill.find(')')].strip()
            else:
                skill_num = skill.strip()
                param = None
            
            # 根据技能编号创建相应的任务
            if skill_num == 'skill01':
                tasks.append({
                    'name': 'generate_mask',
                    'params': {'prompt': param},
                    'delay': 1
                })
            elif skill_num == 'skill02':
                tasks.append({
                    'name': 'detect_grasp_pose',
                    'delay': 1
                })
            elif skill_num == 'skill03':
                tasks.append({
                    'name': 'detect_release_pose',
                    'delay': 1
                })
            elif skill_num == 'skill04':
                tasks.append({
                    'name': 'move_to_target_pose',
                    'delay': 1
                })
            elif skill_num == 'skill05':
                tasks.append({
                    'name': 'grasp',
                    'delay': 1
                })
            elif skill_num == 'skill06':
                tasks.append({
                    'name': 'release',
                    'delay': 1
                })
            elif skill_num == 'skill07':
                tasks.append({
                    'name': 'rotate',
                    'delay': 1
                })
            elif skill_num == 'skill08':
                tasks.append({
                    'name': 'move_home',
                    'delay': 2
                })
            elif skill_num == 'skill09':
                tasks.append({
                    'name': 'pull',
                    'delay': 1
                })
        
        return tasks

def main():
    # 创建客户端实例
    client = PlannerClient()

    # # 图片路径
    # image_path = r"/home/rl/dexVLM/src/grasp/assets/kitchen.jpg"

    # # 捕获图像
    # color_image, _ = client.capture_images()
    # # 获取GPT-4响应
    # requester = GPT4Requester(question_id=4,ignore_former_conversation=False)
    # print("waiting for response...")
    # response = requester.request_gpt4(requester.question, color_image)
    
    # response= "##skill01(the smaller carambola)##skill02##skill04##skill05##skill01(box)##skill03##skill04##skill06##"  
    detect_grasp = "##skill01(carambola)##skill02##skill04##"
    grasp1 = "##skill01(orange)##skill02##skill04##skill05##"
    grasp2 = "##skill01(orange)##skill06##skill02##skill04##skill05##"
    release2 = "##skill01(bowl)##skill03##skill04##skill06##skill08##"
    grasp3 = "##skill01(red apple)##skill02##skill04##skill05##"
    release3 = "##skill01(box inner)##skill03##skill04##skill06##skill08##"
    grasp4 = "##skill01(green apple)##skill02##skill04##skill05##"
    release4 = "##skill01(box inner)##skill03##skill04##skill06##skill08##"

    response = detect_grasp
    # response = "##skill06##" # release
    # response = "##skill09##" # pull
    # response = "##skill08##" # movehome
    # response = release2

    # 解析响应生成任务序列
    tasks = client.parse_skill_sequence(response)
    print("\nGenerated task sequence:")
    for i, task in enumerate(tasks):
        print(f"{i+1}. {task['name']}", end='')
        if 'params' in task:
            print(f" with params: {task['params']}")
        else:
            print()
    
    # 执行任务流程
    print("\nExecuting task sequence...")
    client.execute_pipeline(tasks)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
    except Exception as e:
        print(f"\nError executing pipeline: {e}") 