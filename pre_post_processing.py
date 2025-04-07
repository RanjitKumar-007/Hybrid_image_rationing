"""Importing libraries"""
import numpy as np
import cv2
import rasterio # pylint: disable=import-error
import tifffile # pylint: disable=import-error
from osgeo import gdal
class pre_post_processing:
    def convert_gray_to_rgb(self, image):
        """Conversion of single band image to three band image"""
        if len(image.shape) == 2:
            return np.broadcast_to(image[:, :, np.newaxis], (*image.shape, 3))
        return image
    def calculate_index(self, band1, band2, formula):
        """Generalizing index calculation function"""
        b_1 = band1.astype('float64')
        b_2 = band2.astype('float64')
        return formula(b_1, b_2)
    def process_image(self, bands):
        """NDVI, NDWI, NDSI, and SAVI calculation"""
        ndvi = self.calculate_index(bands['nir'], bands['red'],
        lambda nir, red: np.where((nir + red) == 0., 0, (nir - red) / (nir + red)))
        ndwi = self.calculate_index(bands['green'], bands['nir'],
        lambda green, nir: np.where((green + nir) == 0., 0, (green - nir) / (green + nir)))
        ndsi = self.calculate_index(bands['swir1'], bands['nir'],
        lambda swir1, nir: np.where((swir1 + nir) == 0., 0, (swir1 - nir) / (swir1 + nir)))
        savi = self.calculate_index(bands['nir'], bands['red'],
        lambda nir, red: np.where((nir + red + 0.5) == 0., 0,
                                  (nir - red)*(1 + 0.5) / (nir + red + 0.5)))
        return ndvi, ndwi, ndsi, savi
    def create_masks(self, ndvi, ndwi, ndsi, savi):
        """Creating masks based on thresholds for both images"""
        vegetation_mask = ndvi > 0.1
        water_mask = ndwi > 1
        shadow_mask = ndsi > 1
        soil_mask = savi > 1
        combined_mask = np.logical_or(np.logical_or
                                      (vegetation_mask, water_mask), shadow_mask, soil_mask)
        return combined_mask
    def apply_mask(self, image, mask):
        """Applying combined mask to each image"""
        mask = mask[:, :, np.newaxis]
        return np.where(mask, np.nan, image.astype('float64'))
    def load_optical_tif_ntf_images(self, image_path1, image_path2, apply_preprocessing=False):
        """Load optical TIFF/TIF and NITF/NTF images"""
        if image_path1.lower().endswith(('.tif', '.tiff')):
            def read_image(image_path):
                with rasterio.open(image_path) as src:
                    bands = src.read()
                return bands.transpose(1, 2, 0)
            image1 = read_image(image_path1)
            image2 = read_image(image_path2)
        elif image_path1.lower().endswith(('.nitf', '.ntf')):
            dataset_1 = gdal.Open(image_path1)
            dataset_2 = gdal.Open(image_path2)
            image1 = dataset_1.ReadAsArray().transpose(1, 2, 0)
            image2 = dataset_2.ReadAsArray().transpose(1, 2, 0)
        if image1 is None or image2 is None:
            print(f"Error: Unable to load images {image_path1} or {image_path2}")
            return None, None
        if image1.shape != image2.shape:
            image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0])) # pylint: disable=no-member
        if not apply_preprocessing:
            return image1, image2
        bands1 = {
            'red': image1[:, :, 0], 'green': image1[:, :, 1], 'blue': image1[:, :, 2],
            'nir': image1[:, :, 3], 'swir1': image1[:, :, 4], 'swir2': image1[:, :, 5]
        }
        bands2 = {
            'red': image2[:, :, 0], 'green': image2[:, :, 1], 'blue': image2[:, :, 2],
            'nir': image2[:, :, 3], 'swir1': image2[:, :, 4], 'swir2': image2[:, :, 5]
        }
        ndvi1, ndwi1, ndsi1, savi1 = self.process_image(bands1)
        ndvi2, ndwi2, ndsi2, savi2 = self.process_image(bands2)
        mask1 = self.create_masks(ndvi1, ndwi1, ndsi1, savi1)
        mask2 = self.create_masks(ndvi2, ndwi2, ndsi2, savi2)
        masked_image1 = self.apply_mask(image1, mask1)
        masked_image2 = self.apply_mask(image2, mask2)
        masked_image1 = np.nan_to_num(masked_image1, nan=0, posinf=0, neginf=0)
        masked_image2 = np.nan_to_num(masked_image2, nan=0, posinf=0, neginf=0)
        return masked_image1, masked_image2
    def load_sar_rgb_images(self, image_path1, image_path2):
        """Load one and three channel images"""
        if image_path1.lower().endswith(('.tif', '.tiff')):
            image1 = tifffile.imread(image_path1)
            image2 = tifffile.imread(image_path2)
        elif image_path1.lower().endswith(('.nitf', '.ntf')):
            dataset_1 = gdal.Open(image_path1)
            dataset_2 = gdal.Open(image_path2)
            image1 = dataset_1.ReadAsArray()
            image2 = dataset_2.ReadAsArray()
        elif image_path1.lower().endswith(('.jpg', '.jpeg', '.bmp', '.png')):
            image1 = cv2.imread(image_path1) # pylint: disable=no-member
            image2 = cv2.imread(image_path2) # pylint: disable=no-member
        if image1 is None or image2 is None:
            print(f"Error: Unable to load images {image_path1} or {image_path2}")
            return None, None
        image1 = self.convert_gray_to_rgb(image1)
        image2 = self.convert_gray_to_rgb(image2)
        if image1.shape != image2.shape:
            image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0])) # pylint: disable=no-member
        return image1, image2
    def process_tif(self, input_data):
        """Process TIFF and TIF image array and return it as an RGB array"""
        if isinstance(input_data, np.ndarray):
            if len(input_data.shape) == 3:
                if input_data.shape[0] <= 3:
                    input_data = input_data.transpose(1, 2, 0)
                num_bands = input_data.shape[2]
                is_grayscale = num_bands < 3
                if is_grayscale:
                    red = np.nan_to_num(input_data[:, :, 0], nan=0)
                    green = blue = red
                else:
                    red = np.nan_to_num(input_data[:, :, 0], nan=0)
                    green = np.nan_to_num(input_data[:, :, 1], nan=0)
                    blue = np.nan_to_num(input_data[:, :, 2], nan=0)
            elif len(input_data.shape) == 2:
                is_grayscale = True
                red = np.nan_to_num(input_data, nan=0)
                green = blue = red
            else:
                raise ValueError("Unsupported array shape, expected 2D or 3D array")
            min_val = min(red.min(), green.min(), blue.min())
            max_val = max(red.max(), green.max(), blue.max())
            red = ((red - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            green = ((green - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            blue = ((blue - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            rgb = np.dstack((red, green, blue))
            return rgb, is_grayscale
        else:
            raise ValueError("No TIFF and TIF image array")
    def process_ntf(self, input_data):
        """Process NITF and NTF image array and return it as an RGB array"""
        if isinstance(input_data, np.ndarray):
            if len(input_data.shape) == 3:
                if input_data.shape[0] <= 3:
                    input_data = input_data.transpose(1, 2, 0)
                num_bands = input_data.shape[2]
                is_grayscale = num_bands < 3
                if is_grayscale:
                    red = np.nan_to_num(input_data[:, :, 0], nan=0)
                    green = blue = red
                else:
                    red = np.nan_to_num(input_data[:, :, 0], nan=0)
                    green = np.nan_to_num(input_data[:, :, 1], nan=0)
                    blue = np.nan_to_num(input_data[:, :, 2], nan=0)
            elif len(input_data.shape) == 2:
                is_grayscale = True
                red = np.nan_to_num(input_data, nan=0)
                green = blue = red
            else:
                raise ValueError("Unsupported array shape, expected 2D or 3D array")
            red = np.nan_to_num(red, nan=0)
            green = np.nan_to_num(green, nan=0)
            blue = np.nan_to_num(blue, nan=0)
            min_val = min(red.min(), green.min(), blue.min())
            max_val = max(red.max(), green.max(), blue.max())
            if max_val > min_val:
                red = ((red - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                green = ((green - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                blue = ((blue - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            else:
                red = green = blue = np.zeros_like(red, dtype=np.uint8)
            rgb = np.dstack((red, green, blue))
            return rgb, is_grayscale
        else:
            raise ValueError("No NITF and NTF image array")
    def automatic_brightness_contrast(self, image, clip_hist_percent=2.5):
        """Automatically adjusts brightness and contrast to preserve color properties"""
        if len(image.shape) == 3:
            zero_pixels = np.count_nonzero(np.all(image == 0, axis=2))
            total_pixels = image.shape[0] * image.shape[1]
        else:
            zero_pixels = np.count_nonzero(image == 0)
            total_pixels = image.size
        zero_percentage = zero_pixels / total_pixels * 100
        if zero_percentage < 10:
            clip_hist_percent = 2.0
            contrast_boost = 1.0
            brightness_adjustment = 0.0
            strategy_name = "Low Zeros"
        elif 10 <= zero_percentage < 30:
            clip_hist_percent = 2.3
            contrast_boost = 1.0
            brightness_adjustment = 0.0
            strategy_name = "Medium Zeros"
        elif 30 <= zero_percentage < 50:
            clip_hist_percent = 1.8
            contrast_boost = 1.0
            brightness_adjustment = 0.0
            strategy_name = "High Zeros"
        else:
            clip_hist_percent = 1.8
            contrast_boost = 1.0
            brightness_adjustment = 0.0
            strategy_name = "Extreme Zeros"
        print(f"Selected Strategy: {strategy_name}")
        if len(image.shape) == 3 and image.shape[2] > 3:
            process_image = image[:, :, :3].copy()
        else:
            process_image = image.copy()
        process_image = process_image.astype(np.float32)
        global_alpha, global_beta = 1.0, 0.0
        channel_count = min(3, process_image.shape[2]) if len(process_image.shape) == 3 else 1
        for i in range(channel_count):
            channel = process_image[:, :, i] if len(process_image.shape) == 3 else process_image
            non_zero_mask = channel > 0
            hist, _ = np.histogram(channel[non_zero_mask].flatten(), bins=256, range=[0, 256])
            cdf = hist.cumsum()
            total_non_zero_pixels = channel[non_zero_mask].size
            clip_value = clip_hist_percent * total_non_zero_pixels / 100.0
            minimum = np.argmax(cdf > clip_value)
            maximum = np.argmax(cdf > cdf[-1] - clip_value)
            if maximum > minimum:
                alpha = (255.0 / (maximum - minimum)) * contrast_boost
                beta = -minimum * alpha + brightness_adjustment
                channel_adjusted = channel * alpha + beta
                channel[non_zero_mask] = np.clip(channel_adjusted[non_zero_mask], 0, 255)
                global_alpha *= alpha
                global_beta += beta
                darken_threshold = 50
                lighten_threshold = 200
                channel[non_zero_mask & (channel < darken_threshold)] *= 0.9
                mask = non_zero_mask & (channel > lighten_threshold)
                channel[mask] += (255 - channel[mask]) * 0.2
                zero_pixel_factor = 0.7 + (0.1 * (brightness_adjustment / 20))
                channel[~non_zero_mask] *= zero_pixel_factor
                if len(process_image.shape) == 3:
                    process_image[:, :, i] = np.clip(channel, 0, 255)
                else:
                    process_image = np.clip(channel, 0, 255)
        enhanced_image = np.clip(process_image, 0, 255).astype(np.uint8)
        return enhanced_image, global_alpha, global_beta
    def overlay_image(self, change_map_resized, image):
        """Overlay change map on input images"""
        _, binary_mask = cv2.threshold(change_map_resized, 200, 255, cv2.THRESH_BINARY)  # pylint: disable=no-member
        green_mask = np.zeros_like(image)
        indices = np.where(binary_mask == 255)
        green_mask[indices[0], indices[1], :] = [0, 255, 0]
        overlay = cv2.add(image, green_mask)  # pylint: disable=no-member
        return overlay
    def save_nitf_ntf_image(self, output_path, overlay, ref_image_path):
        """Save overlay as a NITF and NTF image"""
        dataset = gdal.Open(ref_image_path)
        driver = gdal.GetDriverByName("NITF")
        out_dataset = driver.Create(output_path, dataset.RasterXSize,
                                    dataset.RasterYSize, 3, gdal.GDT_Byte)
        for i in range(3):
            out_dataset.GetRasterBand(i + 1).WriteArray(overlay[:, :, i])
        out_dataset.FlushCache()
    def hybrid_image_rationing(self, scale_1, scale_2):
        """Image rationing with differencing and edge enhancement"""
        scale_1 = scale_1.astype(np.float32)
        scale_2 = scale_2.astype(np.float32)
        epsilon = 1e-5
        ratio_image = np.divide(scale_1 + epsilon, scale_2 + epsilon)
        ratio_image = np.clip(ratio_image, 0.1, 10)
        log_ratio_image = np.log1p(ratio_image)
        diff_image = np.abs(scale_1 - scale_2)
        edge_enhanced = cv2.Laplacian(diff_image.astype(np.uint8), cv2.CV_64F) # pylint: disable=no-member
        edge_enhanced = np.abs(edge_enhanced)
        fused_result = log_ratio_image + edge_enhanced + diff_image
        normalized_image = ((fused_result - fused_result.min()) /
                            (fused_result.max() - fused_result.min()) * 255).astype(np.uint8)
        return normalized_image
    def change_detection(self, ratio_img, threshold):
        """Change detection using image rationing"""
        _, change_mask = cv2.threshold(ratio_img, threshold, 255, cv2.THRESH_BINARY) # pylint: disable=no-member
        return change_mask
