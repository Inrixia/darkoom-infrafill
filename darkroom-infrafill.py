import os
import cv2
import numpy as np

# global variables
max_value_16bit = 65535

def normalize(image, low=2, high=99):
    float_image = image.astype('float32')

    # Calculate low and high percentiles
    p_low, p_high = np.percentile(float_image, (low, high))
    
    # Create a mask for values within the percentile range
    mask = ((float_image > p_low) & (float_image < p_high)).astype(np.uint8)

    # Rescale the intensity of the image using the mask
    rescaled_image = cv2.normalize(float_image, None, alpha=0, beta=max_value_16bit, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F, mask=mask)

    # Normalize to 16-bit max value, adjusting with the new max of rescaled image
    normalized = (rescaled_image * max_value_16bit / rescaled_image.max()).astype('uint16')
    return normalized

def dust_mask(infrared_image, positive_image):
    r_channel, _, _ = cv2.split(positive_image)

    normalized_infrared = normalize(infrared_image, 2, 99)
    normalized_red = normalize(r_channel, 2, 99)

    c = normalized_infrared / ((normalized_red.astype('float') + 1) / (max_value_16bit + 1))
    divided = c * (c < max_value_16bit) + max_value_16bit * np.ones(np.shape(c)) * (c > max_value_16bit)

    ret, inverted = cv2.threshold(divided, (max_value_16bit-max_value_16bit/1), max_value_16bit, cv2.THRESH_BINARY_INV)
    
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.dilate(inverted, kernel, iterations = 1)
    mask = max_value_16bit - mask

    return mask

def process_image(input_path, output_folder, save_mask=False):
    layered, src = cv2.imreadmulti(input_path, [], cv2.IMREAD_UNCHANGED)

    if len(src) < 2:
        print(f"{input_path} - SKIPPING! Must contain at least two layers.")
        return
    elif src[0].dtype != np.uint16 or src[1].dtype != np.uint16:
        print(f"{input_path} - SKIPPING! Must be in 16-bit format.")
        return
    else:
        ir = src[2]
        img = src[0]

    print(f"{input_path} - Generating dust mask...")
    mask = dust_mask(ir, img)

    filename = os.path.basename(os.path.splitext(input_path)[0])

    if save_mask:
        output_path_mask = os.path.join(output_folder, f"{filename}_mask.tif")
        print(f"{input_path} - Saving mask to {output_path_mask}")
        cv2.imwrite("./test/mask.tif", mask.astype(np.uint16), params=(cv2.IMWRITE_TIFF_COMPRESSION, 5))

    print(f"{input_path} - Combining dust mask with positive image...")
    out = np.dstack((img, mask))

    output_path = os.path.join(output_folder, f"{filename}_clean.tif")
    print(f"{input_path} - Saving image to {filename}")
    cv2.imwrite(output_path, out.astype(np.uint16), params=(cv2.IMWRITE_TIFF_COMPRESSION, 5))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process .tif images.")
    parser.add_argument("input", help="Input folder or file path")
    parser.add_argument("output", help="Output folder path")
    parser.add_argument("save-mask", action="store_true", help="Save dust mask")
    args = parser.parse_args()

    input_path = args.input
    output_folder = os.path.normpath(args.output)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if os.path.isfile(input_path):
        if not input_path.endswith(".tif"):
            print(f"Invalid input path. {input_path} is not a .tif file. Please provide a valid .tif file or a folder containing .tif files.")
            exit(1)
        process_image(input_path, output_folder)
    elif os.path.isdir(input_path):
        for filename in os.listdir(input_path):
            if filename.endswith(".tif"):
                process_image(f"{input_path}/{filename}", output_folder)
    else:
        print("Invalid input path.")