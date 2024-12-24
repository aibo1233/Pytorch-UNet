import cv2
import numpy as np
import os
from PIL import Image
from matplotlib.path import Path


def binarize_rgb_image(image, threshold=128):
    """
    对 RGB 图像进行二值化处理，输出结果也是 RGB 图像。

    参数:
        image (np.ndarray): 输入的 RGB 图像，形状为 (H, W, 3)。
        threshold (int): 二值化的阈值，默认值为 128。

    返回:
        np.ndarray: 二值化后的 RGB 图像，形状为 (H, W, 3)。
    """
    # 将 RGB 图像转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # 创建二值化掩码
    _, binary_mask = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)

    # 将掩码扩展为三通道，直接应用于 RGB 图像
    binary_rgb_image = cv2.merge([binary_mask, binary_mask, binary_mask])

    return binary_rgb_image


# # 示例
# # 创建一个简单的 RGB 图像
# rgb_image = np.zeros((10, 10, 3), dtype=np.uint8)
# rgb_image[2:8, 2:8] = [100, 150, 200]  # 中间部分为颜色值 (100, 150, 200)

# # 对图像进行二值化
# binary_rgb_image = binarize_rgb_image(rgb_image, threshold=128)

# # 显示原始图像和二值化后的图像
# plt.subplot(1, 2, 1)
# plt.imshow(rgb_image)
# plt.title("Original RGB Image")
# plt.axis("off")

# plt.subplot(1, 2, 2)
# plt.imshow(binary_rgb_image)
# plt.title("Binarized RGB Image")
# plt.axis("off")

# plt.show()


def increment_polygon_area(matrix, points):
    """
    对一个矩阵中由 points 定义的多边形区域的值加 1。

    参数:
        matrix (np.ndarray): 输入的二维矩阵。
        points (list of tuple): 定义多边形的顶点，格式为 [(x1, y1), (x2, y2), ...]。

    返回:
        np.ndarray: 更新后的矩阵。
    """
    # 将顶点转换为适合 OpenCV 的格式
    polygon_points = np.array([points], dtype=np.int32)

    # 创建一个与输入矩阵形状相同的掩码
    mask = np.zeros_like(matrix, dtype=np.uint8)

    # 在掩码上绘制多边形区域
    cv2.fillPoly(mask, polygon_points, 1)

    # 将掩码区域加 1
    matrix += mask

    return matrix


# # 示例
# matrix = np.zeros((10, 10), dtype=np.int32)  # 创建一个 10x10 的零矩阵
# polygon_points = [(2, 2), (2, 7), (7, 7), (7, 2)]  # 定义多边形的顶点
# updated_matrix = increment_polygon_area(matrix, polygon_points)

# print(updated_matrix)


def draw_polygon_on_matrix(matrix, points, value=255):
    """
    在矩阵上根据顶点列表绘制多边形，并填充指定值。

    参数:
        matrix (np.ndarray): 输入的二维矩阵。
        points (list of tuple): 定义多边形的顶点，格式为 [(x1, y1), (x2, y2), ...]。
        value (int or float): 填充多边形的值。

    返回:
        np.ndarray: 绘制后的矩阵。
    """
    # 将顶点转换为适合 OpenCV 的格式
    points = np.array(points, dtype=np.int32)

    # 创建一个与输入矩阵形状相同的空白矩阵
    output = np.copy(matrix)

    # 在矩阵上绘制多边形
    cv2.fillPoly(output, [points], value)

    return output


# # 示例
# matrix = np.zeros((10, 10), dtype=np.uint8)  # 创建一个 10x10 的零矩阵
# polygon_points = [(2, 2), (2, 7), (7, 7), (7, 2)]  # 定义多边形的顶点
# updated_matrix = draw_polygon_on_matrix(matrix, polygon_points, value=255)

# print(updated_matrix)


def crop_by_points(image, points, crop_width, crop_height):
    """
    从图像中根据四个点裁剪并仿射变换为矩形区域。
    
    :param image: 输入图像 (numpy array)
    :param points: 四个点的坐标 [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    :return: 仿射变换后的裁剪图像
    """
    # 确保点按顺时针或逆时针顺序排列
    points = np.array(points, dtype="float32")

    # 目标矩形的宽和高 (可通过点间距离估算)
    width = int(max(
        np.linalg.norm(points[0] - points[1]),
        np.linalg.norm(points[2] - points[3])
    ))
    height = int(max(
        np.linalg.norm(points[0] - points[3]),
        np.linalg.norm(points[1] - points[2])
    ))

    # 目标矩形的点 (左上、右上、右下、左下)
    dst_points = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")

    # 计算仿射变换矩阵
    print(type(points[0][0]))
    print(points)
    print(type(dst_points[0][0]))
    matrix = cv2.getPerspectiveTransform(points, dst_points)

    # 进行透视变换
    cropped_image = cv2.warpPerspective(image, matrix, (width, height))

    # 输出的图片统一为1920*1080
    resized_image = cv2.resize(cropped_image, (crop_width, crop_height))

    return resized_image


def calculate_overlap_area(original_points, transformed_points, image_width, image_height):
    """计算重叠面积的比例"""
    # 将原始和变换后的点转化为多边形
    original_points = np.array(original_points, dtype=np.int32)
    transformed_points = np.array(transformed_points, dtype=np.int32)

    # 创建空白图像，用于绘制多边形
    mask_original = np.zeros((image_height, image_width), dtype=np.uint8)
    mask_transformed = np.zeros((image_height, image_width), dtype=np.uint8)

    # 绘制多边形区域
    cv2.fillPoly(mask_original, [original_points], 255)
    cv2.fillPoly(mask_transformed, [transformed_points], 255)

    # 计算重叠区域
    overlap = cv2.bitwise_and(mask_original, mask_transformed)
    overlap_area = np.sum(overlap)  # 重叠区域的像素数

    # 计算原图面积
    # original_area = np.sum(mask_original)  # 原图多边形的像素数
    original_area = max(np.sum(mask_original),np.sum(mask_transformed))  #两个图形中面积的更大值

    # 返回重叠面积与原图面积的比例
    return overlap_area / original_area if original_area > 0 else 0


def transform_points_with_metadata(points, img_width, img_height):
    """
    对四个点进行平移、旋转和缩放变换，并输出包含类型、程度和方向信息的结果。
    
    :param points: 原始四个点坐标 [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    :param img_width: 图像宽度
    :param img_height: 图像高度
    :return: 包含变换结果及元数据的列表
    """
    results = []
    points = np.array(points, dtype=np.float32)
    
    # 图像中心
    center = np.array([img_width / 2, img_height / 2])
    mean_length = (img_width + img_height) / 2

    # 平移
    translation_offsets = [0.02, 0.04, 0.06, 0.08, 0.1]  # 平移百分比
    directions = {
        "up": [0, -1],
        "down": [0, 1],
        "left": [-1, 0],
        "right": [1, 0],
        "up_left": [-1, -1],
        "up_right": [1, -1],
        "down_left": [-1, 1],
        "down_right": [1, 1]
    }
    for offset in translation_offsets:
        d = offset * mean_length
        for direction, vector in directions.items():
            translated_points = points + np.array(vector) * d
            overlap_ratio = calculate_overlap_area(points,translated_points,img_height,img_width)
            results.append({
                "type": "translation",
                "magnitude": f"{int(offset * 100):02}%",
                "direction": direction,
                "overlap_ratio":overlap_ratio,
                "points": [(int(x), int(y)) for x, y in translated_points]
            })
    
    # 旋转
    rotation_angles = [4,8,12,16,20,24,28,32,36,40]
    for angle in rotation_angles:
        for direction in ["cw", "ccw"]:
            theta = np.radians(angle if direction == "cw" else -angle)
            rotation_matrix = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]
            ])
            rotated_points = (points - center) @ rotation_matrix.T + center
            overlap_ratio = calculate_overlap_area(points,rotated_points,img_height,img_width)
            results.append({
                "type": "rotation",
                "magnitude": f"{angle:02}degree",
                "direction": "clockwise" if direction == "cw" else "counterclockwise",
                "overlap_ratio":overlap_ratio,
                "points": [(int(x), int(y)) for x, y in rotated_points]
            })
    
    # 缩放
    scaling_factors = [1.02, 1.04, 1.06, 1.08, 1.1,1.12,1.14,1.16,1.18,1.2]
    for factor in scaling_factors:
        for mode in ["enlarge", "shrink"]:
            scale = factor if mode == "enlarge" else 1 / factor
            scaled_points = (points - center) * scale + center
            overlap_ratio = calculate_overlap_area(points,scaled_points,img_height,img_width)
            results.append({
                "type": "scaling",
                "magnitude": f"{int((factor - 1) * 100):02}%" if mode == "enlarge" else f"{int((1 - 1 / factor) * 100):02}%",
                "direction": mode,
                "overlap_ratio":overlap_ratio,
                "points": [(int(x), int(y)) for x, y in scaled_points]
            })
    
    return results


def save_image_with_info(image, info, folder_path):
    """
    将图像保存到指定文件夹中，以 `info` 的内容生成文件名。
    
    :param image: 图像内容（Pillow Image 对象）
    :param info: 包含类型、变换程度、方向和点信息的字典
    :param folder_path: 保存文件的目标文件夹路径
    """
    # 确保目标文件夹存在
    os.makedirs(folder_path, exist_ok=True)
    
    # 根据 info 生成文件名
    filename = f"{info['raw_img_name']}_{info['type']}_{info['direction']}_{info['magnitude']}_{info['overlap_ratio']}.png"
    
    # 去除文件名中不合法的字符
    filename = filename.replace(" ", "").replace("(", "").replace(")", "").replace(",", "_").replace(":", "")
    
    # 拼接完整路径
    file_path = os.path.join(folder_path, filename)
    
    # 保存图像
    # image.save(file_path)
    cv2.imwrite(file_path,image)
    print(f"Image saved to: {file_path}")



def crop_center_with_points(image, crop_width, crop_height):
    """
    从图像中心裁剪指定尺寸的区域，并返回裁剪区域的四个角点坐标。
    
    :param image: 输入图像 (numpy array)
    :param crop_width: 裁剪区域的宽度
    :param crop_height: 裁剪区域的高度
    :return: 裁剪后的图像和四个角点坐标
    """
    # 获取图像尺寸
    h, w = image.shape[:2]
    
    # 计算中心点
    center_x, center_y = w // 2, h // 2
    
    # 计算裁剪区域的边界
    x_start = center_x - crop_width // 2
    x_end = center_x + crop_width // 2
    y_start = center_y - crop_height // 2
    y_end = center_y + crop_height // 2
    
    # 裁剪图像
    cropped_image = image[y_start:y_end, x_start:x_end]
    
    # 计算四个角点的坐标
    points = [
        (x_start, y_start),  # 左上角
        (x_end, y_start),    # 右上角
        (x_end, y_end),      # 右下角
        (x_start, y_end)     # 左下角
    ]
    
    return cropped_image, points



# 示例调用
if __name__ == "__main__":
    # 检查文件夹是否存在
    dir_img_raw = 'data/imgs'
    dir_img_transform = 'data/imgs_transform'
    dir_mask_raw = 'data/masks'
    dir_mask_transform = 'data/masks_transform'
    if not os.path.exists(dir_img_transform):
        os.mkdir(dir_img_transform)
    if not os.path.exists(dir_mask_transform):
        os.mkdir(dir_mask_transform)


    
    # 创建一个示例大图像 (4000x4000)
    raw_image_dir = 'raw_img/'
    for pic in os.listdir(raw_image_dir):
        pic_path = os.path.join(raw_image_dir,pic)
        image = cv2.imread(pic_path)
        # image = cv2.imread("wallhaven-5gjwo1_3000.jpg")
        # 获取图像尺寸
        img_height, img_width = image.shape[:2]

        # 裁剪中心的 1920x1080 区域
        cropped, points = crop_center_with_points(image, 1920, 1080)
        

        # 对四个点进行变换
        transformed_results = transform_points_with_metadata(points, img_width, img_height)
        # print(transformed_results)

        # 截取图像
        for item in transformed_results:
            print(item)
            points_temp = item['points']
            item['raw_img_name'] = pic.split('.')[0]


            # 生成mask
            mask = np.zeros((img_height, img_width), dtype=np.uint8)
            # 在原图截取区域内+1
            mask = increment_polygon_area(mask,points)
            # 在变换图截取区域内+1
            mask = increment_polygon_area(mask,points_temp)


            # 非公共区域为0，公共区域为255
            mask[mask != 2] = 0
            mask[mask ==2]=255
            mask_rgb = np.stack([mask] * 3, axis=-1)


            # # 可视化
            # dir_visual = 'data/visual'
            # img_visual = np.zeros((img_height, img_width), dtype=np.uint8)
            # img_visual = draw_polygon_on_matrix(img_visual,points)
            # img_visual = draw_polygon_on_matrix(img_visual,points_temp)
            # img_visual_rgb = np.stack([img_visual] * 3, axis=-1)
            # # 保存visual图
            # save_image_with_info(img_visual_rgb, item, folder_path=dir_visual)

            # mask_原图区域
            mask_raw = crop_by_points(mask_rgb,points,1920,1080)
            # 二值化，因为变换后边缘会模糊
            mask_raw = binarize_rgb_image(mask_raw)
            # mask_变换图区域
            mask_transform = crop_by_points(mask_rgb,points_temp,1920, 1080)
            # 二值化，因为变换后边缘会模糊
            mask_transform = binarize_rgb_image(mask_transform)
            # 生成变换图
            cropped_image = crop_by_points(image,points_temp,1920, 1080)
            
            

            # 保存原图
            save_image_with_info(cropped, item, folder_path=dir_img_raw)
            # 保存变换图
            save_image_with_info(cropped_image, item, folder_path=dir_img_transform)
            # 保存原图mask
            save_image_with_info(mask_raw, item, folder_path=dir_mask_raw)
            # 保存变换图mask
            save_image_with_info(mask_transform, item, folder_path=dir_mask_transform)
   
