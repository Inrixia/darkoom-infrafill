import os
import cv2
import numpy as np
import threading

# global variables
max_value_16bit = 65535

def normalize(image, low=3, high=99):
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

def display_img(image, name="Img"):
    target_width = 2560
    target_height = 1440

    # Calculate new dimensions while maintaining aspect ratio
    aspect_ratio = image.shape[1] / image.shape[0]
    if image.shape[0] > target_height or image.shape[1] > target_width:
        if aspect_ratio > 16 / 9:  # Wide image
            target_height = int(target_width / aspect_ratio)
        else:  # Tall image
            target_width = int(target_height * aspect_ratio)

    cv2.imshow(name, cv2.rotate(cv2.resize(image, (target_width, target_height)), cv2.ROTATE_90_CLOCKWISE))

def dust_mask(infrared_image, positive_image):
    r_channel, g, _ = cv2.split(positive_image)

    cleared = normalize(r_channel - infrared_image)

    _, mask = cv2.threshold(cleared, max_value_16bit*0.12, max_value_16bit, cv2.THRESH_BINARY)

    kernel_size = 3
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Perform dilation
    mask = cv2.dilate(mask, kernel, iterations=5)

    # Invert so image previews dont show the entire image as transparent
    mask = max_value_16bit - mask    

    # display_img(mask, "mask")
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return mask


def process_image(input_path, output_folder, save_mask=False):
    layered, src = cv2.imreadmulti(input_path, [], cv2.IMREAD_UNCHANGED)
    filename = os.path.basename(os.path.splitext(input_path)[0])

    if len(src) < 2:
        print(f"{filename} - SKIPPING! Must contain at least two layers.")
        return
    elif src[0].dtype != np.uint16 or src[1].dtype != np.uint16:
        print(f"{filename} - SKIPPING! Must be in 16-bit format.")
        return
    else:
        ir = src[2]
        img = src[0]

    print(f"{filename} - Generating dust mask...")
    mask = dust_mask(ir, img)

    if save_mask:
        output_path_mask = os.path.join(output_folder, f"{filename}_mask.tif")
        print(f"{filename} - Saving mask to {output_path_mask}")
        cv2.imwrite(output_path_mask, mask.astype(np.uint16), params=(cv2.IMWRITE_TIFF_COMPRESSION, 5))

    out = np.dstack((img, mask))

    output_path = os.path.join(output_folder, f"{filename}_clean.tif")
    print(f"{filename} - Saving image to {filename}")
    cv2.imwrite(output_path, out.astype(np.uint16), params=(cv2.IMWRITE_TIFF_COMPRESSION, 5))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process .tif images.")
    parser.add_argument("input", help="Input folder or file path")
    parser.add_argument("output", help="Output folder path")
    parser.add_argument("--save-mask", action="store_true", help="Save dust mask")
    args = parser.parse_args()

    input_path = args.input
    output_folder = os.path.normpath(args.output)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if os.path.isfile(input_path):
        if not input_path.endswith(".tif"):
            print(f"Invalid input path. {input_path} is not a .tif file. Please provide a valid .tif file or a folder containing .tif files.")
            exit(1)
        process_image(input_path, output_folder, args.save_mask)
    elif os.path.isdir(input_path):
        threads = []
        for filename in os.listdir(input_path):
            if filename.endswith(".tif"):
                thread = threading.Thread(target=process_image, args=(os.path.join(input_path, filename), output_folder, args.save_mask))
                thread.start()
                threads.append(thread)
        
        for thread in threads:
            thread.join()
    else:
        print("Invalid input path.")