import gc
from osgeo import gdal
import os
import numpy as np

def split_large_tiff_skip_black_white(input_file, output_dir, tile_size):
    """
    将超大 TIFF 图像切分为指定大小的块，并跳过包含 [0, 0, 0] 或 [255, 255, 255] 的块。

    Args:
        input_file (str): 输入的超大 TIFF 文件路径。
        output_dir (str): 切分后图像保存的目录。
        tile_size (int): 每块的宽度和高度（像素）。
    """
    # 打开大图像文件
    dataset = gdal.Open(input_file)
    if dataset is None:
        raise ValueError("无法打开输入 TIFF 文件")
    
    img_width = dataset.RasterXSize
    img_height = dataset.RasterYSize
    num_bands = dataset.RasterCount

    print(f"图像大小：{img_width}×{img_height}, 波段数：{num_bands}")
    no_data_values = [dataset.GetRasterBand(i + 1).GetNoDataValue() for i in range(num_bands)]
    
    os.makedirs(output_dir, exist_ok=True)
    
    for y in range(0, img_height, tile_size):
        for x in range(0, img_width, tile_size):
            x_size = min(tile_size, img_width - x)
            y_size = min(tile_size, img_height - y)
            tile_data = dataset.ReadAsArray(x, y, x_size, y_size)

            if tile_data is None:
                continue

            # 检查是否为全 NoData 块
            is_all_nodata = True
            for band_index in range(num_bands):
                if np.any(tile_data[band_index] != no_data_values[band_index]):
                    is_all_nodata = False
                    break
            if is_all_nodata:
                print(f"跳过全 NoData 块，位置：({x}, {y})")
                continue

            # 检查是否包含 [0, 0, 0] 或 [255, 255, 255]
            if num_bands == 3:
                black_pixel = (tile_data[0] == 0) & (tile_data[1] == 0) & (tile_data[2] == 0)
                white_pixel = (tile_data[0] == 255) & (tile_data[1] == 255) & (tile_data[2] == 255)
                if np.any(black_pixel) or np.any(white_pixel):
                    print(f"跳过包含黑色或白色像素的块，位置：({x}, {y})")
                    continue

            # 保存当前块
            output_file = os.path.join(output_dir, f"tile_{x}_{y}.tiff")
            driver = gdal.GetDriverByName("GTiff")
            out_dataset = driver.Create(output_file, x_size, y_size, num_bands, dataset.GetRasterBand(1).DataType)

            for band in range(num_bands):
                out_band = out_dataset.GetRasterBand(band + 1)
                out_band.WriteArray(tile_data[band])
                out_band.FlushCache()

            geo_transform = dataset.GetGeoTransform()
            out_dataset.SetGeoTransform([
                geo_transform[0] + x * geo_transform[1],
                geo_transform[1],
                geo_transform[2],
                geo_transform[3] + y * geo_transform[5],
                geo_transform[4],
                geo_transform[5],
            ])
            out_dataset.SetProjection(dataset.GetProjection())
            out_dataset.FlushCache()
            
            # 清理资源
            tile_data = None
            out_dataset = None
            gc.collect()

            print(f"已保存 {output_file}")

    print("切分完成！")

# 使用示例
input_file = "../WHU_DATASET/1.the whole aerial image.tif"  # 输入的超大 TIFF 图像路径
output_dir = "./raw_img_whu_dataset/"  # 保存切分图像的目录
tile_size = 3000                  # 切分块的大小

split_large_tiff_skip_black_white(input_file, output_dir, tile_size)
