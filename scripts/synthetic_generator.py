import os
import sys
import cv2
import numpy as np
from PIL import Image
import glob
import random
import datetime
from tqdm import tqdm

# Add parent directory to path for importing tools
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tools.contour_tools import *
from tqdm import tqdm

"""
grid 에서 영역 선택하는거 수정 필요 : 리스트 새로만들어서 큰 의미 x
원래 객체 사이즈 안넘치게 하려고 했던건데
-> 일단 grid 나눠놓고 영역 선택한 뒤에 객체 사이즈 빼줘야할듯?
이미지 사이즈는 이미지 크기에 랜덤하게 reseize하는게 아니라
"""
root_path = "./dataset/"
save_path = "./save_img/"

# class_list=['Camera', 'Lens', 'Phone', 'face']
class_list = ["Camera", "Phone"]
split_list = ["train", "val", "test"]
now = datetime.datetime.now()
month = now.strftime("%m")
day = now.strftime("%d")

# Update the save_path with the month and day
# save_path_img = f'./save_img/result_{month}_{day}_'
save_path_img = f"./dataset/save_img/"
# save_path_txt = f'./save_txt/image_{}'

# Get a list of all cropped images in the "crop_img" folder
random.seed("1234")
cropped_images_list = glob.glob(root_path + "contour_image/**/*.*", recursive=True)
random.shuffle(cropped_images_list)
copy_crop = cropped_images_list
original_array = np.array(copy_crop)

# 배열의 모양을 (3, 80)으로 재구성
reshaped_contour_image = original_array.reshape(5, 186)

print(np.shape(reshaped_contour_image))
print(len(reshaped_contour_image[0]))
# Load the background image
background_img = np.array(Image.open(root_path + "bg.jpg"))
background_list = glob.glob(root_path + "office_background/*.*")
# Choose 5 random cropped images
# random_cropped_images = random.sample(cropped_images_list,10)
bg_area = 1280 * 720
min_area = int(0.04 * bg_area)
max_area = int(0.13 * bg_area)
print(
    "Contour images : {}\nBackground images : {}".format(
        len(cropped_images_list), len(background_list)
    )
)
for idx2, bg_img in enumerate(background_list):
    # break
    if idx2 < 192:
        continue
    print("BG...{}/{}".format(idx2 + 1, len(background_list)))
    copy_crop_list = cropped_images_list

    # Process each randomly chosen cropped image
    # background_img = np.array(Image.open(bg_img))

    # bg_area = background_img.shape[0] * background_img.shape[1]
    # min_area = int(0.04 * bg_area)
    # max_area = int(0.15 * bg_area)
    draw_guideline(
        background_img, background_img.shape[0], background_img.shape[1], 10, 10
    )

    bounding_boxes = []
    # img_result = background_img[:, :, :3].copy()

    # for idx, cropped_image_path in enumerate(tqdm(cropped_images_list)):
    for idx, cropped_image_path in enumerate(tqdm(reshaped_contour_image[idx2 % 5])):
        background_img = []
        background_img = np.array(Image.open(bg_img))
        img_result = background_img[:, :, :3].copy()

        # Read the cropped image and find the contour
        imgc_alpha = cv2.imread(cropped_image_path, cv2.IMREAD_UNCHANGED)
        # print(cropped_image_path)
        # angle = random.randint(-45,45)
        angle = 0
        imgc_alpha = rotate_image(imgc_alpha, angle)
        contour_np, cropped_image = find_contour(imgc_alpha)

        # Calculate the area of the cropped image
        cropped_area = contour_np.shape[0] * contour_np.shape[1]

        # Calculate the target area size for the cropped image based on the background area
        target_area = random.randint(min_area, max_area)

        # target_area = min_area
        aspect_ratio = cropped_image.shape[1] / cropped_image.shape[0]
        desired_width = int(np.sqrt(target_area * aspect_ratio))
        desired_height = int(
            cropped_image.shape[0] * (desired_width / cropped_image.shape[1])
        )

        # Calculate the scaling factor to resize the cropped image while maintaining its aspect ratio
        # cv2.imshow('1',cropped_image)
        # Resize the cropped image
        resized_cropped_image = cv2.resize(
            cropped_image, (desired_width, desired_height)
        )

        contour_np, cropped_image = find_contour(resized_cropped_image)
        # cv2.imshow('2',cropped_image)

        # Randomly choose a location to place the cropped image on the background
        ## 최대한 겹치치 않는 선에서 랜덤하게 뿌려주려고
        ## 배경을 일정 크기로 나누고, 그 공간을 리스트에서 랜덤하게 뽑아서 사용
        ## 이렇게 하면 안됨. 리스트 계속 새로 만들어서 제대로 못막아줌
        # 이제 어차피 한장만 넣기로 해서 상관없음.
        rand_width_list = [
            i
            for i in range(
                0,
                int(background_img.shape[1] - contour_np.shape[1]),
                int(background_img.shape[1] - contour_np.shape[1]) // 20,
            )
        ]
        # step_value = max(1, (background_img.shape[1] - contour_np.shape[1]) // 20)
        # rand_width_list = [i for i in range(0, int(background_img.shape[1] - contour_np.shape[1]), step_value)]
        # print('width list',rand_width_list)
        # step_value = max(1, (background_img.shape[0] - contour_np.shape[0]) // 10)
        # rand_height_list = [i for i in range(0, int(background_img.shape[0] - contour_np.shape[0]), step_value)]

        # print('height list',rand_height_list)
        try:
            rand_height_list = [
                i
                for i in range(
                    0,
                    int(background_img.shape[0] - contour_np.shape[0]),
                    int(background_img.shape[0] - contour_np.shape[0]) // 10,
                )
            ]
        except:
            print(int(background_img.shape[0] - contour_np.shape[0]))
            print(background_img.shape[0], contour_np.shape[0])
        # if rand_width_list:
        x = random.sample(rand_width_list, 1)[0]
        # else:
        #     x=0
        # if rand_height_list:
        try:
            y = random.sample(rand_height_list, 1)[0]
        except:
            print(cropped_image_path)
            print(rand_height_list)
            print("bg", bg_area)
            print("max", max_area)
            print("de_w*de_h", desired_width * desired_height)

        # else:
        #     y=0
        # if x==0 or y==0:
        #     print('x,y',x,y)

        name = os.path.basename(cropped_image_path).split("_")[0]
        bounding_box_info, check = create_bbox(
            background_img,
            x,
            y,
            cropped_image.shape[1],
            cropped_image.shape[0],
            class_list.index(name),
        )
        if check == 0:
            print("back image", idx2, "_", idx, ".jpg")
            print("w,h is more than 1", cropped_image_path)
            print("w, h", bounding_box_info["width"], bounding_box_info["height"])
        bounding_boxes.append(bounding_box_info)
        # Get the cropped image with an alpha channel

        # Calculate the alpha mask from the contour
        my_alpha = contour_np[:, :] / 255.0
        # Create a copy of the background image to overlay the cropped image on
        # img_result = background_img[:, :, :3].copy()

        # Overlay the cropped image onto the background
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)

        overlay_image_alpha(img_result, cropped_image[:, :, :3], x, y, my_alpha)
        background_img = img_result
        text = str(desired_width) + "," + str(desired_height)
        # cv2.putText(img_result, text,(x,y),2,2,(255,255,255))
        # showing_img = cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB)

        # Save the result
        # print('idxxxxxxx',idx2, int(len(background_list)*0.7))
        # break
        if idx2 < int(len(background_list) * 0.7):
            split_name = "train"
        elif idx2 < int(len(background_list) * 0.85):
            split_name = "valid"
        else:
            split_name = "test"
        save_syn_path = (
            save_path_img + split_name + "/images/{}_{}".format(idx2, idx) + ".jpg"
        )
        save_label_path = (
            save_path_img + split_name + "/labels/{}_{}".format(idx2, idx) + ".txt"
        )
        # print('save file :', save_syn_path)
        with open(save_label_path, "w") as txt_file:
            # for bounding_box in bounding_boxes:
            txt_file.write(
                f"{bounding_box_info['class']} {bounding_box_info['x_center']} {bounding_box_info['y_center']} {bounding_box_info['width']} {bounding_box_info['height']}\n"
            )
        # cv2.imshow('asdf',img_result)
        # cv2.waitKey(0)
        # img_result_rgb = cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB)
        # cv2.imshow('asdf',img_result_rgb)
        # cv2.waitKey(0)
        Image.fromarray(img_result).save(save_syn_path)
