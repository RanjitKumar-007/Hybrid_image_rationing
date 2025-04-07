"""Importing libraries"""
import time
import os
import cv2
import rasterio # pylint: disable=import-error
import tifffile # pylint: disable=import-error
from osgeo import gdal
from pre_post_processing import pre_post_processing
obj = pre_post_processing()
def run_change_detection(image_path1, image_path2):
    """Main execution block"""
    start_time = time.time()
    file_ext1 = os.path.splitext(image_path1)[-1].lower()
    file_ext2 = os.path.splitext(image_path2)[-1].lower()
    if file_ext1 != file_ext2:
        print("Error: Image formats do not match")
        return
    if file_ext1 in ['.jpg', '.jpeg', '.bmp', '.png']:
        image1, image2 = obj.load_sar_rgb_images(image_path1, image_path2)
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY) # pylint: disable=no-member
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY) # pylint: disable=no-member
    elif file_ext1 in ['.tif', '.tiff']:
        with rasterio.open(image_path1) as src1:
            bands1 = src1.read()
        with rasterio.open(image_path2) as src2:
            bands2 = src2.read()
        num_bands1 = bands1.shape[0]
        num_bands2 = bands2.shape[0]
        if num_bands1 == 1 and num_bands2 == 1:
            print("Detected: single-band image")
            image1, image2 = obj.load_sar_rgb_images(image_path1, image_path2)
            gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY) # pylint: disable=no-member
            gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY) # pylint: disable=no-member
        else:
            print("Detected: multi-band image")
            user_input = int(input("1: Without preprocessing, 2: With preprocessing: "))
            image1, image2 = obj.load_optical_tif_ntf_images(image_path1,
                                                     image_path2, apply_preprocessing=False)
            rgb_image1_tif, is_grayscale = obj.process_tif(image1)
            if not is_grayscale:
                rgb_image1_tif, _, _ = obj.automatic_brightness_contrast(rgb_image1_tif)
            rgb_image2_tif, is_grayscale = obj.process_tif(image2)
            if not is_grayscale:
                rgb_image2_tif, _, _ = obj.automatic_brightness_contrast(rgb_image2_tif)
            if user_input == 2:
                masked_image1, masked_image2 = obj.load_optical_tif_ntf_images(
                  image_path1, image_path2, apply_preprocessing=True)
                rgb_mask_image1_tif, is_grayscale = obj.process_tif(masked_image1)
                if not is_grayscale:
                    rgb_mask_image1_tif, _, _ = obj.automatic_brightness_contrast(rgb_mask_image1_tif)
                rgb_mask_image2_tif, is_grayscale = obj.process_tif(masked_image2)
                if not is_grayscale:
                    rgb_mask_image2_tif, _, _ = obj.automatic_brightness_contrast(rgb_mask_image2_tif)
                gray1 = cv2.cvtColor(rgb_mask_image1_tif, cv2.COLOR_BGR2GRAY) # pylint: disable=no-member
                gray2 = cv2.cvtColor(rgb_mask_image2_tif, cv2.COLOR_BGR2GRAY) # pylint: disable=no-member
            else:
                gray1 = cv2.cvtColor(rgb_image1_tif, cv2.COLOR_BGR2GRAY) # pylint: disable=no-member
                gray2 = cv2.cvtColor(rgb_image2_tif, cv2.COLOR_BGR2GRAY) # pylint: disable=no-member
    elif file_ext1 in ['.ntf', '.nitf']:
        dataset_1 = gdal.Open(image_path1)
        dataset_2 = gdal.Open(image_path2)
        num_bands1 = dataset_1.RasterCount
        num_bands2 = dataset_2.RasterCount
        if num_bands1 == 1 and num_bands2 == 1:
            print("Detected: single-band image")
            image1, image2 = obj.load_sar_rgb_images(image_path1, image_path2)
            gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)  # pylint: disable=no-member
            gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)  # pylint: disable=no-member
        else:
            print("Detected: multi-band image")
            user_input = int(input("1: Without preprocessing, 2: With preprocessing: "))
            image1, image2 = obj.load_optical_tif_ntf_images(image_path1,
                                                     image_path2, apply_preprocessing=False)
            rgb_image1_ntf, is_grayscale = obj.process_ntf(image1)
            if not is_grayscale:
                rgb_image1_ntf, _, _ = obj.automatic_brightness_contrast(rgb_image1_ntf)
            rgb_image2_ntf, is_grayscale = obj.process_ntf(image2)
            if not is_grayscale:
                rgb_image2_ntf, _, _ = obj.automatic_brightness_contrast(rgb_image2_ntf)
            if user_input == 2:
                masked_image1, masked_image2 = obj.load_optical_tif_ntf_images(
                  image_path1, image_path2, apply_preprocessing=True)
                rgb_mask_image1_ntf, is_grayscale = obj.process_ntf(masked_image1)
                if not is_grayscale:
                    rgb_mask_image1_ntf, _, _ = obj.automatic_brightness_contrast(rgb_mask_image1_ntf)
                rgb_mask_image2_ntf, is_grayscale = obj.process_ntf(masked_image2)
                if not is_grayscale:
                    rgb_mask_image2_ntf, _, _ = obj.automatic_brightness_contrast(rgb_mask_image2_ntf)
                gray1 = cv2.cvtColor(rgb_mask_image1_ntf, cv2.COLOR_BGR2GRAY) # pylint: disable=no-member
                gray2 = cv2.cvtColor(rgb_mask_image2_ntf, cv2.COLOR_BGR2GRAY) # pylint: disable=no-member
            else:
                gray1 = cv2.cvtColor(rgb_image1_ntf, cv2.COLOR_BGR2GRAY) # pylint: disable=no-member
                gray2 = cv2.cvtColor(rgb_image2_ntf, cv2.COLOR_BGR2GRAY) # pylint: disable=no-member
    if gray1.shape != gray2.shape:
        gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0])) # pylint: disable=no-member
    threshold_val = int(input("Threshold (0-255): ")) # User can vary threshold (preferred 40-60)
    rationing = obj.hybrid_image_rationing(gray1, gray2)
    change_map = obj.change_detection(rationing, threshold_val)
    end_time = time.time()
    print("Computational time in seconds:", end_time - start_time)
    output_path = None
    if all(path.lower().endswith(('.tif', '.tiff')) for path in (image_path1, image_path2)):
        with rasterio.open(image_path1) as src:
            num_bands = src.count
        output_path = 'Output_image.tif'
        if num_bands > 3:
            overlay = obj.overlay_image(change_map, rgb_image1_tif)
        else:
            overlay = obj.overlay_image(change_map, image1)
        tifffile.imwrite(output_path, overlay)
    elif all(path.lower().endswith(('.nitf')) for path in (image_path1, image_path2)):
        dataset = gdal.Open(image_path1, gdal.GA_ReadOnly)
        num_bands = dataset.RasterCount
        output_path = 'Output_image.nitf'
        if num_bands > 3:
            overlay = obj.overlay_image(change_map, rgb_image1_ntf)
        else:
            overlay = obj.overlay_image(change_map, image1)
        obj.save_nitf_ntf_image(output_path, overlay, image_path1)
    elif all(path.lower().endswith(('.ntf')) for path in (image_path1, image_path2)):
        dataset = gdal.Open(image_path1, gdal.GA_ReadOnly)
        num_bands = dataset.RasterCount
        output_path = 'Output_image.ntf'
        if num_bands > 3:
            overlay = obj.overlay_image(change_map, rgb_image1_ntf)
        else:
            overlay = obj.overlay_image(change_map, image1)
        obj.save_nitf_ntf_image(output_path, overlay, image_path1)
    else:
        overlay = obj.overlay_image(change_map, image1)
        output_path = 'Output_image.png'
        cv2.imwrite(output_path, overlay) # pylint: disable=no-member
    print(f'Output saved to {output_path}')
image_path1 = 'air5.png'  # Path of Image_1
image_path2 = 'air6.png'  # Path of Image_2
run_change_detection(image_path1, image_path2)
