import os
import cv2
import glob

# Specify the path to the image folder
image_folder_path = './dataset/office_background'
image_folder_list = glob.glob(image_folder_path + '/**/*.*', recursive=True)
print(len(image_folder_list))

# Calculate the total width and height
total_width = 0
total_height = 0
valid_image_files = []

for image_file in image_folder_list:
    try:
        img = cv2.imread(image_file)
        if img is not None:
            height, width, _ = img.shape
            total_width += width
            total_height += height
            valid_image_files.append(image_file)
        else:
            print(f"Invalid image: {image_file}")
    except Exception as e:
        print(f"Error processing image {image_file}: {e}")

# Calculate the average width and height
average_width = total_width // len(valid_image_files)
average_height = total_height // len(valid_image_files)

# Specify the desired image size
desired_width = 1280
desired_height = 720

# Resize valid images to the desired size
for image_file in valid_image_files:
    img = cv2.imread(image_file)
    
    # Resize the image to the desired size
    resized_img = cv2.resize(img, (desired_width, desired_height))
    
    # Save the resized image back to the same file
    cv2.imwrite(image_file, resized_img)

print(f"Average Width: {average_width}")
print(f"Average Height: {average_height}")
