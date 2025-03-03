"""
    This script is used to test the graspnet API.
    It will load the model and detect the grasp pose of the object in the image.
    To run this script, you need to run demo.sh.
"""

import os
import argparse
import torch
import numpy as np
import open3d as o3d
from PIL import Image
import time
from gsnet import AnyGrasp

from graspnetAPI import GraspGroup

import pyrealsense2 as rs
import cv2

import ultralytics
from collections import defaultdict
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path',default="log/checkpoint_detection.tar", help='Model checkpoint path')
parser.add_argument('--max_gripper_width', type=float, default=0.1, help='Maximum gripper width (<=0.1m)')
parser.add_argument('--gripper_height', type=float, default=0.08, help='Gripper height')
parser.add_argument('--top_down_grasp', default=True, action='store_true', help='Output top-down grasps.')
parser.add_argument('--debug', default=True, action='store_true', help='Enable debug mode')
cfgs = parser.parse_args()
cfgs.max_gripper_width = max(0, min(0.2, cfgs.max_gripper_width))

def show_point_cloud(color_image, depth_image, fx, fy, cx, cy, scale):
    """显示RGB-D点云"""
    # 创建点云对象
    pcd = o3d.geometry.PointCloud()
    
    # 生成点云坐标
    xmap, ymap = np.arange(depth_image.shape[1]), np.arange(depth_image.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)
    
    points_z = depth_image / scale
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z
    
    # 过滤无效点
    mask = (points_z > 0) & (points_z < 1)
    points = np.stack([points_x, points_y, points_z], axis=-1)
    points = points[mask]
    
    # 获取对应的颜色
    colors = color_image.astype(np.float32) / 255.0
    colors = colors[mask]
    
    # 设置点云的点和颜色
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 创建坐标系
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    
    # 显示点云
    o3d.visualization.draw_geometries([pcd, coordinate_frame])

def grasp_box():
    try:
        
        anygrasp = AnyGrasp(cfgs)
        anygrasp.load_net()
        print("anygrasp loaded")

        # 从本地读取 RGB 图像
        colors = cv2.imread('../assets/color_image.jpg')
        
        # 从本地读取深度数据
        depths = np.load('../assets/depth_image_data.npy')

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depths, alpha=0.4), cv2.COLORMAP_JET)
        colors = colors.astype(np.float32) / 255.0

        # 使用原始相机内参
        fx, fy = 608.013, 608.161
        cx, cy = 318.828, 241.382
        scale = 1000.0

        # 显示原始点云
        # print("Showing original point cloud...")
        # show_point_cloud(colors, depths, fx, fy, cx, cy, scale)

        # 设置工作空间
        xmin, xmax = -0.35, 0.35
        ymin, ymax = -0.35, 0.35
        zmin, zmax = 0, 1.0
        lims = [xmin, xmax, ymin, ymax, zmin, zmax]

        # 获取点云
        xmap, ymap = np.arange(depths.shape[1]), np.arange(depths.shape[0])
        xmap, ymap = np.meshgrid(xmap, ymap)
        points_z = depths / scale
        points_x = (xmap - cx) / fx * points_z
        points_y = (ymap - cy) / fy * points_z

        mask = (points_z > 0) & (points_z < 1)
        points = np.stack([points_x, points_y, points_z], axis=-1)
        points = points[mask].astype(np.float32)
        colors = colors[mask].astype(np.float32)

        print("querying grasp")

        # 使用 torch.no_grad() 减少内存使用
        with torch.no_grad():
            gg, cloud = anygrasp.get_grasp(points, colors, lims=lims, 
                                         apply_object_mask=True, 
                                         dense_grasp=True, 
                                         collision_detection=True)

        if len(gg) == 0:
            print('No Grasp detected after collision detection!')
            return None
        else:
            gg = gg.nms().sort_by_score()
            gg_pick = gg[0:20]
            print(gg_pick.scores)
            print('grasp score:', gg_pick[0].score)
            print('gg_pick :', gg_pick[0])

            # 构造返回字典
            best_grasp = gg_pick[0]
            grasp_result = {
                'translation': best_grasp.translation,
                'rotation_matrix': best_grasp.rotation_matrix,
                'score': float(best_grasp.score)
            }

        print("finished")
        cv2.waitKey(0)

        # visualization
        if cfgs.debug:
            trans_mat = np.array([[-1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
            cloud.transform(trans_mat)
            grippers = gg.to_open3d_geometry_list()
            for gripper in grippers:
                gripper.transform(trans_mat)
            o3d.visualization.draw_geometries([*grippers, cloud])
            o3d.visualization.draw_geometries([grippers[0], cloud])

        # 返回抓取位姿字典
        return grasp_result

    except RuntimeError as e:
        if "out of memory" in str(e):
            print("GPU内存不足，尝试清理内存...")
            torch.cuda.empty_cache()
            return None
        else:
            raise e

def capture_images(visualize=True):

    # 初始化管道，不断初始化后续需要优化！！！
    pipeline = rs.pipeline()
    # 配置流
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # 深度流
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # RGB 流
    # 启动管道
    pipeline.start(config)

    # 创建对齐对象，目标是 RGB 流
    align_to = rs.stream.color
    align = rs.align(align_to)

    try:
        # 等待帧并获取其
        frames = pipeline.wait_for_frames()

        # 对齐深度帧到 RGB
        aligned_frames = align.process(frames)

        # 获取对齐后的深度图像和 RGB 图像
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        depth_profile = aligned_depth_frame.get_profile()
        # print("parameters:", depth_profile)
        dvsprofile = rs.video_stream_profile(depth_profile)
        depth_intrin = dvsprofile.get_intrinsics()
        print("depth_intrin", depth_intrin)
        # 将深度图像转换为 NumPy 数组
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        print("depth_image", depth_image)
        # 
        class_ids, class_labels, boxes, confidences, bg_color_img, bg_depth_img = process_images(color_image, depth_image)

        print("class id:", class_ids)
        if visualize:
            # 可视化深度图像（需要转换为伪彩色）
            # print("bg_depth_img", bg_depth_img)
            bg_depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(bg_depth_img, alpha=0.4), cv2.COLORMAP_JET)
            if class_ids is not None:
                for box, class_id, confidence in zip(boxes, class_ids, confidences):
                    x, y, w, h = box
                    # Convert xywh to xyxy
                    x1 = int(x - w / 2)
                    y1 = int(y - h / 2)
                    x2 = int(x + w / 2)
                    y2 = int(y + h / 2)
                    
                    # Draw the bounding box on the image
                    cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box with thickness 2
                    # Prepare the label text
                    label = f"{class_labels[int(class_id)]}: {confidence:.2f}"
                    # Draw the label at the top-left corner of the bounding box
                    cv2.putText(color_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
                    
                    # print the depth pixels within bbox
                    box_depth = bg_depth_img[y1:y2, x1:x2]
                    if class_id > 6:
                        print("Depth values within box:")
                        print(box_depth)
                        # np.savetxt(f'depths_data_class_{int(class_id)}.txt', box_depth, fmt='%.2f')
            # 显示 RGB 和深度图像
            cv2.imshow('bg_depth_image', bg_depth_img)
            cv2.imshow('bg_depth_colormap', bg_depth_colormap)
            cv2.imshow('Color Image', color_image)
            cv2.imshow('bg_color_img', bg_color_img)
            cv2.waitKey(0)

    finally:
        # 停止管道
        pipeline.stop()
        cv2.destroyAllWindows()

    return bg_depth_img, bg_color_img  # 返回处理后的深度图像和 RGB 图像
    # return depth_image, color_image  # 返回处理后的深度图像和 RGB 图像
# set the pixels out of bbox as mean_value

def process_images(rgb_image, depth_image):
    model = YOLO("./yolo11_hitv2.pt")
    # rgb_image = cv2.imread("examples/pic/01.jpg")
    # 创建一个 Annotator 对象
    annotator = Annotator(rgb_image, line_width=1)

    # 执行对象跟踪
    results = model.predict(source=rgb_image, conf=0.25, show=False)

    # 提取边界框、object类别id、类别标签、mask
    boxes = results[0].boxes.xywh.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy()
    class_labels = results[0].names
    confidences = results[0].boxes.conf.cpu().numpy()

    # 创建一个全为 True 的掩码
    mask = np.ones(rgb_image.shape[:2], dtype=bool)
    # 遍历每个 box，将其区域设置为 False
    confidences = results[0].boxes.conf.cpu().numpy()
    for box, confidence, class_id in zip(boxes, confidences, class_ids):
        if confidence > 0.5 and class_id > 6:
            x, y, w, h = box
            # x1 = int(x - w / 2)
            # y1 = int(y - h / 2)
            # x2 = int(x + w / 2)
            # y2 = int(y + h / 2)

            # 计算扩大10%后的宽度和高度,考虑边缘深度信息
            w_expanded = w * 1.6
            h_expanded = h * 1.6

            # 计算新的边界框坐标
            x1 = int(x - w_expanded / 2)
            y1 = int(y - h_expanded / 2)
            x2 = int(x + w_expanded / 2)
            y2 = int(y + h_expanded / 2)
            
            # 确保坐标不超出图像边界
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(rgb_image.shape[1], x2)
            y2 = min(rgb_image.shape[0], y2)
            mask[y1:y2, x1:x2] = False

    # print("mask:", mask) 
    # 计算 boxes 以外的像素的均值
    background_mean = rgb_image[mask].mean(axis=0)
    bg_color_img = np.copy(rgb_image)
    bg_depth_img = np.copy(depth_image)
    bg_color_img[mask] = background_mean
    bg_depth_img[mask] = 10000

    # # 返回所需的结果
    return class_ids, class_labels, boxes, confidences, bg_color_img, bg_depth_img

# 示例调用
# rgb_image, depth_image = capture_images(visualize=False)
# masks, class_ids, class_labels, boxes, boxes_shape = process_images(rgb_image, depth_image)

def detect_grasp_box(rgb_image, depth_image):
    """检测抓取位姿，接受完整的 RGB 图像和深度图像（边界框外的深度值已设为2m）"""
    try:
        anygrasp = AnyGrasp(cfgs)
        anygrasp.load_net()
        print("anygrasp loaded")

        # 使用传入的 RGB 图像
        colors = rgb_image
        
        # 使用传入的深度数据
        depths = depth_image

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depths, alpha=0.4), cv2.COLORMAP_JET)
        colors = colors.astype(np.float32) / 255.0

        # 使用原始相机内参
        fx, fy = 608.013, 608.161
        cx, cy = 318.828, 241.382
        scale = 1000.0

        # 设置工作空间
        xmin, xmax = -0.35, 0.35
        ymin, ymax = -0.35, 0.35
        zmin, zmax = 0, 1.0
        lims = [xmin, xmax, ymin, ymax, zmin, zmax]

        # 获取点云
        xmap, ymap = np.arange(depths.shape[1]), np.arange(depths.shape[0])
        xmap, ymap = np.meshgrid(xmap, ymap)
        points_z = depths / scale
        points_x = (xmap - cx) / fx * points_z
        points_y = (ymap - cy) / fy * points_z

        # 只保留深度在有效范围内的点（自动过滤掉2m以外的点）
        mask = (points_z > 0) & (points_z < 1)
        points = np.stack([points_x, points_y, points_z], axis=-1)
        points = points[mask].astype(np.float32)
        colors = colors[mask].astype(np.float32)

        print("querying grasp")
        time.sleep(7)

        # 使用 torch.no_grad() 减少内存使用
        with torch.no_grad():
            gg, cloud = anygrasp.get_grasp(points, colors, lims=lims, 
                                         apply_object_mask=True, 
                                         dense_grasp=True,                            # 是否密集抓取
                                         collision_detection=True)

        if len(gg) == 0:
            print('No Grasp detected after collision detection!')
            return None
        else:
            gg = gg.nms().sort_by_score()
            gg_pick = gg[0:20]
            print(gg_pick.scores)
            print('grasp score:', gg_pick[0].score)
            print('gg_pick :', gg_pick[0])

            # 构造返回字典
            best_grasp = gg_pick[0]
            grasp_result = {
                'translation': best_grasp.translation,
                'rotation_matrix': best_grasp.rotation_matrix,
                'score': float(best_grasp.score)
            }

        print("finished")
        cv2.waitKey(0)

        # visualization
        if cfgs.debug:
            trans_mat = np.array([[-1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
            cloud.transform(trans_mat)
            grippers = gg.to_open3d_geometry_list()
            for gripper in grippers:
                gripper.transform(trans_mat)
            o3d.visualization.draw_geometries([*grippers, cloud])
            o3d.visualization.draw_geometries([grippers[0], cloud])

        # 清理 GPU 缓存
        torch.cuda.empty_cache()
        # 返回抓取位姿字典
        return grasp_result

    except RuntimeError as e:
        if "out of memory" in str(e):
            print("GPU内存不足，尝试清理内存...")
            torch.cuda.empty_cache()
            return None
        else:
            raise e

def visualize_grasp_pose(rgb_image, depth_image):
    """可视化抓取位姿"""
    try:
        anygrasp = AnyGrasp(cfgs)
        anygrasp.load_net()
        print("anygrasp loaded")

        # 使用传入的 RGB 图像
        colors = rgb_image
        
        # 使用传入的深度数据
        depths = depth_image

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depths, alpha=0.4), cv2.COLORMAP_JET)
        colors = colors.astype(np.float32) / 255.0

        # 使用原始相机内参
        fx, fy = 608.013, 608.161
        cx, cy = 318.828, 241.382
        scale = 1000.0

        # 设置工作空间
        xmin, xmax = -0.35, 0.35
        ymin, ymax = -0.35, 0.35
        zmin, zmax = 0, 1.0
        lims = [xmin, xmax, ymin, ymax, zmin, zmax]

        # 获取点云
        xmap, ymap = np.arange(depths.shape[1]), np.arange(depths.shape[0])
        xmap, ymap = np.meshgrid(xmap, ymap)
        points_z = depths / scale
        points_x = (xmap - cx) / fx * points_z
        points_y = (ymap - cy) / fy * points_z

        # 只保留深度在有效范围内的点（自动过滤掉2m以外的点）
        mask = (points_z > 0) & (points_z < 1)
        points = np.stack([points_x, points_y, points_z], axis=-1)
        points = points[mask].astype(np.float32)
        colors = colors[mask].astype(np.float32)

        print("querying grasp")
        time.sleep(7)

        # 使用 torch.no_grad() 减少内存使用
        with torch.no_grad():
            gg, cloud = anygrasp.get_grasp(points, colors, lims=lims, 
                                         apply_object_mask=True, 
                                         dense_grasp=True,                            # 是否密集抓取
                                         collision_detection=True)

        if len(gg) == 0:
            print('No Grasp detected after collision detection!')
            return None
        else:
            gg = gg.nms().sort_by_score()
            gg_pick = gg[0:20]
            print(gg_pick.scores)
            print('grasp score:', gg_pick[0].score)
            print('gg_pick :', gg_pick[0])

            # 构造返回字典
            best_grasp = gg_pick[0]
            grasp_result = {
                'translation': best_grasp.translation,
                'rotation_matrix': best_grasp.rotation_matrix,
                'score': float(best_grasp.score)
            }

        print("finished")
        cv2.waitKey(0)

        # visualization
        if cfgs.debug:
            trans_mat = np.array([[-1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
            cloud.transform(trans_mat)
            grippers = gg.to_open3d_geometry_list()
            for gripper in grippers:
                gripper.transform(trans_mat)
            o3d.visualization.draw_geometries([*grippers, cloud])
            o3d.visualization.draw_geometries([grippers[0], cloud])

        # 清理 GPU 缓存
        torch.cuda.empty_cache()
        # 返回抓取位姿字典
        return grasp_result

    except RuntimeError as e:
        if "out of memory" in str(e):
            print("GPU内存不足，尝试清理内存...")
            torch.cuda.empty_cache()
            return None
        else:
            raise e
    



if __name__ == '__main__':
    # capture_images()

    # grasp_box()

    # 从本地读取 RGB 图像
    colors = cv2.imread('../assets/color_image.jpg')
    # 从本地读取深度数据
    depths = np.load('../assets/depth_image_data.npy')

    # 调用 detect_grasp_box 函数
    grasp_result = detect_grasp_box(colors, depths)  # 传入 RGB 图像和深度图像
