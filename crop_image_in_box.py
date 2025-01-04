import os
import rasterio
import shapefile
from rasterio.windows import Window
from rasterio.transform import Affine

def read_shapefile(shp_path):
    with shapefile.Reader(shp_path) as shp:
        shapes = shp.shapes()
        if len(shapes) != 1:
            raise ValueError("Shapefile must contain exactly one shape (rectangle).")
        return shapes[0].bbox  # Returns (minx, miny, maxx, maxy)

def cut_tiff_to_tiles(tiff_path, bbox, output_dir, tile_size=3000):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with rasterio.open(tiff_path) as src:
        minx, miny, maxx, maxy = bbox
        bounds_transform = src.transform
        col_start, row_start = ~bounds_transform * (minx, maxy)
        col_end, row_end = ~bounds_transform * (maxx, miny)

        col_start, row_start = int(col_start), int(row_start)
        col_end, row_end = int(col_end), int(row_end)

        for row_off in range(row_start, row_end, tile_size):
            for col_off in range(col_start, col_end, tile_size):
                window = Window(col_off, row_off, tile_size, tile_size)
                transform = src.window_transform(window)
                tile_array = src.read(window=window)

                output_path = os.path.join(
                    output_dir, f"tile_{row_off}_{col_off}.tif"
                )
                with rasterio.open(
                    output_path,
                    "w",
                    driver="GTiff",
                    height=tile_array.shape[1],
                    width=tile_array.shape[2],
                    count=src.count,
                    dtype=src.dtypes[0],
                    crs=src.crs,
                    transform=transform,
                ) as dst:
                    dst.write(tile_array)

if __name__ == "__main__":
    tiff_path = "../WHU_DATASET/1.the whole aerial image.tif"  # 替换为您的 TIFF 图像路径
    shp_path = "../WHU_DATASET/test.shp"  # 替换为您的 Shapefile 路径
    output_dir = "./raw_img_whu_dataset_check/"  # 替换为您的输出文件夹路径

    bbox = read_shapefile(shp_path)
    cut_tiff_to_tiles(tiff_path, bbox, output_dir)
