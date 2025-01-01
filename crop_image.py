from PIL import Image
import os

def crop_image(input_path, output_path,crop_regions):
    """
    裁剪5000x5000图片为四张3000x3000的图片。

    :param input_path: 输入图片的路径。
    :param output_path: 输出图片的文件夹路径。
    """
    # 确保输出路径存在
    os.makedirs(output_path, exist_ok=True)

    raw_name = os.path.splitext(os.path.basename(input_path))[0]

    # 打开图片
    with Image.open(input_path) as img:
        # 检查图片尺寸是否为5000x5000
        if img.size != (5000, 5000):
            raise ValueError("输入图片的尺寸必须为5000x5000。")



        # 裁剪并保存图片
        for name, box in crop_regions.items():
            cropped_img = img.crop(box)
            cropped_img.save(os.path.join(output_path, f"{raw_name}_{name}.png"))

# 示例用法
# 定义裁剪区域
crop_regions = {
    "top_left": (0, 0, 3000, 3000),
    "top_right": (2000, 0, 5000, 3000),
    "bottom_left": (0, 2000, 3000, 5000),
    "bottom_right": (2000, 2000, 5000, 5000),
}
# crop_regions = {
#     "01": (0, 0, 2000, 1000),
#     "02": (2000, 0, 4000, 1000),
#     "03": (0, 1000, 2000, 2000),
#     "04": (2000, 1000, 4000, 2000),
#     "05": (0, 2000, 2000, 3000),
#     "06": (2000, 2000, 4000, 3000),
#     "07": (0, 3000, 2000, 4000),
#     "08": (2000, 3000, 4000, 4000),
#     "09": (0, 4000, 2000, 5000),
#     "10": (2000, 4000, 4000, 5000),
# }
# crop_image("bellingham1.tif", "output_folder",crop_regions)


# 遍历
image_dir_train = '../AerialImageDataset/train/images'
image_dir_test = '../AerialImageDataset/test/images'
dataset_img_dir = 'raw_img'

for img in os.listdir(image_dir_train):
    input_path = os.path.join(image_dir_train,img)
    crop_image(input_path,dataset_img_dir,crop_regions)
for img in os.listdir(image_dir_test):
    input_path = os.path.join(image_dir_test,img)
    crop_image(input_path,dataset_img_dir,crop_regions)


# ground_truth_dir = 'raw_mask'
# dataset_mask_dir = 'data/masks'
# for mask in os.listdir(ground_truth_dir):
#     input_path = os.path.join(ground_truth_dir,mask)
#     crop_image(input_path,dataset_mask_dir,crop_regions)
