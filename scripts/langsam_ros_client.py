#!/usr/bin/env python3

import rospy
from grasp.srv import MaskGenerate
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def show_images(original_image, masks, result_image):
    """显示原图、mask和结果图"""
    n_masks = len(masks)
    if n_masks == 0:
        return
    
    # 计算总共需要的子图数量（原图 + masks + 结果图）
    total_plots = n_masks + 2
    n_cols = min(3, total_plots)
    n_rows = (total_plots + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=(5*n_cols, 5*n_rows))
    
    # 显示原图
    plt.subplot(n_rows, n_cols, 1)
    plt.imshow(original_image)
    plt.title('Original Image')
    plt.axis('off')
    
    # 显示每个mask
    for i, mask in enumerate(masks, start=2):
        plt.subplot(n_rows, n_cols, i)
        plt.imshow(mask, cmap='gray')
        plt.title(f'Mask {i-1}')
        plt.axis('off')
    
    # 显示结果图
    plt.subplot(n_rows, n_cols, total_plots)
    plt.imshow(result_image.astype(np.uint8))  # 确保结果图像是uint8类型
    plt.title('Result')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def resize_image_height_700(image):
    """调整图像大小，保持宽高比，高度设为700像素"""
    width, height = image.size
    if height <= 700:
        return image
    scale = 700.0 / height
    new_width = int(width * scale)
    return image.resize((new_width, 700), Image.LANCZOS)

def main():
    try:
        # 初始化ROS节点
        rospy.init_node('test_ros_client', anonymous=True)
        print("Node initialized")
        
        # 设置测试参数
        image_path = "../assets/color_image.jpg"
        # text_prompt = "the left carambola on the table"
        text_prompt = "orange"
        box_threshold = 0.3
        text_threshold = 0.25
        
        # 打印请求参数
        print("\nSending request:")
        print(f"  Image: {image_path}")
        print(f"  Prompt: {text_prompt}")
        print(f"  Box threshold: {box_threshold}")
        print(f"  Text threshold: {text_threshold}")
        
        # 等待服务
        print("\nWaiting for service 'generate_mask'...")
        rospy.wait_for_service('generate_mask')
        generate_mask = rospy.ServiceProxy('generate_mask', MaskGenerate)
        
        # 调用服务
        print("Calling service...")
        response = generate_mask(image_path, text_prompt, box_threshold, text_threshold)
        
        if response.masks:
            print(f"\nReceived {len(response.masks)} masks:")
            
            # 加载原始图像并调整大小
            image = Image.open(image_path).convert("RGB")
            # image = resize_image_height_700(image)
            image_array = np.array(image)
            
            # 直接使用 masks
            masks = [np.array(mask.data).reshape(mask.shape) for mask in response.masks]
            
            # 打印每个mask的信息
            for i, mask in enumerate(response.masks):
                print(f"\nMask {i+1}:")
                print(f"  Label: {mask.label}")
                print(f"  Score: {mask.score:.3f}")
                print(f"  Box: [{', '.join(f'{x:.2f}' for x in mask.box)}]")
                # 保存每个mask为png文件
                mask_data = np.array(mask.data).reshape(mask.shape)
                mask_image = Image.fromarray((mask_data * 255).astype(np.uint8))  # 转换为uint8类型
                mask_image.save(f'../assets/outputs/mask_{i+1}.png')  # 保存为PNG文件
            
            # 将所有mask相加并扩展维度以匹配RGB通道
            combined_mask = np.zeros_like(masks[0], dtype=float)
            for mask in masks:
                combined_mask += mask
            combined_mask = np.clip(combined_mask, 0, 1)
            combined_mask = combined_mask[..., np.newaxis]  # 添加通道维度
            
            # 将mask应用到原图（保持RGB值范围）
            result_image = image_array * combined_mask

            
            # 显示所有图像
            show_images(image_array, masks, result_image)
            
        else:
            print("No masks received")
            
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 