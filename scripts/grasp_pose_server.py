#!/usr/bin/env python
import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge
from grasp.srv import GraspPose, GraspPoseResponse
from demo import AnyGrasp
import argparse
import torch
import open3d as o3d

class GraspPoseServer:
    def __init__(self):
        # 初始化ROS节点
        rospy.init_node('grasp_pose_server')
        
        # 创建参数解析器
        parser = argparse.ArgumentParser()
        parser.add_argument('--checkpoint_path', default="log/checkpoint_detection.tar")
        parser.add_argument('--max_gripper_width', type=float, default=0.1)
        parser.add_argument('--gripper_height', type=float, default=0.08)
        parser.add_argument('--top_down_grasp', default=True, action='store_true')
        parser.add_argument('--debug', default=True, action='store_true')    # 是否可视化抓取点云图
        self.cfgs = parser.parse_args([])
        
        # 初始化AnyGrasp模型
        self.anygrasp = AnyGrasp(self.cfgs)
        self.anygrasp.load_net()
        rospy.loginfo("AnyGrasp model loaded successfully")
        
        # 创建CV桥接器
        self.bridge = CvBridge()
        
        # 创建服务
        self.service = rospy.Service('detect_grasp_pose', GraspPose, self.handle_grasp_pose)
        rospy.loginfo("Grasp pose server is ready")

    def handle_grasp_pose(self, req):
        try:
            # 将ROS图像消息转换为OpenCV格式
            rgb_image = self.bridge.imgmsg_to_cv2(req.rgb_image, "bgr8")
            depth_image = self.bridge.imgmsg_to_cv2(req.depth_image)
            print("rgb_image: ", rgb_image.shape)
            print("depth_image: ", depth_image.shape)
            
            # 相机参数
            fx, fy = 608.013, 608.161
            cx, cy = 318.828, 241.382
            scale = 1000.0
            
            # 工作空间限制，单位：米
            lims = [-1, 1, -1, 1, 0, 1.0]
            
            # 处理图像
            colors = rgb_image.astype(np.float32) / 255.0
            depths = depth_image
            
            # 生成点云
            xmap, ymap = np.arange(depths.shape[1]), np.arange(depths.shape[0])
            xmap, ymap = np.meshgrid(xmap, ymap)
            points_z = depths / scale
            points_x = (xmap - cx) / fx * points_z
            points_y = (ymap - cy) / fy * points_z
            
            mask = (points_z > 0) & (points_z < 1)
            points = np.stack([points_x, points_y, points_z], axis=-1)
            points = points[mask].astype(np.float32)
            colors = colors[mask].astype(np.float32)
            
            # 使用AnyGrasp获取抓取姿态
            rospy.loginfo("grasp pose calculation started")
            with torch.no_grad():
                gg, _ = self.anygrasp.get_grasp(
                    points, colors, lims=lims,
                    apply_object_mask=True,
                    dense_grasp=False,
                    collision_detection=True
                )
            rospy.loginfo("pose calculation completed")
            
            response = GraspPoseResponse()
            
            if len(gg) == 0:
                rospy.logwarn('No grasp detected after collision detection!')
                response.success = False
                return response
            
            # 获取最佳抓取
            gg = gg.nms().sort_by_score()
            best_grasp = gg[0]
            
            # 可视化最佳抓取
            if self.cfgs.debug:
                # 创建点云对象
                cloud = o3d.geometry.PointCloud()
                cloud.points = o3d.utility.Vector3dVector(points)
                cloud.colors = o3d.utility.Vector3dVector(colors)
                
                # 应用坐标系变换
                trans_mat = np.array([[-1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
                cloud.transform(trans_mat)
                
                # 获取抓取器几何体
                best_gripper = gg[0:10].to_open3d_geometry_list()
                for gripper in best_gripper:
                    gripper.transform(trans_mat)
                
                # 显示点云和抓取器
                o3d.visualization.draw_geometries([*best_gripper, cloud])
                o3d.visualization.draw_geometries([best_gripper[0], cloud])
            
            # 填充响应
            response.translation = best_grasp.translation.tolist()
            response.rotation_matrix = best_grasp.rotation_matrix.flatten().tolist()
            response.score = float(best_grasp.score)
            response.success = True
            
            # 清理GPU缓存
            torch.cuda.empty_cache()
            
            rospy.loginfo("request processed")
            print("\n")
            return response
            
        except Exception as e:
            rospy.logerr(f"Error in grasp pose server: {str(e)}")
            response = GraspPoseResponse()
            response.success = False
            return response

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        server = GraspPoseServer()
        server.run()
    except rospy.ROSInterruptException:
        pass 