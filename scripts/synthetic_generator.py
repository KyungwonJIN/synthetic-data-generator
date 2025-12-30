import os
import sys
import cv2
import numpy as np
from PIL import Image
import glob
import random
import datetime
import argparse
from tkinter import filedialog, messagebox
import tkinter as tk
from tqdm import tqdm

# Add parent directory to path for importing tools
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tools.contour_tools import *

"""
Synthetic Data Generator
합성 데이터 생성 도구 - rembg로 처리된 객체를 배경 이미지에 합성하여 YOLO 형식 라벨 생성
"""


def get_user_inputs():
    """사용자로부터 입력을 받는 함수"""
    parser = argparse.ArgumentParser(
        description="Synthetic Data Generator - 합성 데이터 생성 도구"
    )
    parser.add_argument(
        "--contour-folder",
        type=str,
        help="Contour 이미지가 있는 폴더 경로 (예: ./dataset/contour_image)",
    )
    parser.add_argument(
        "--background-folder",
        type=str,
        help="배경 이미지가 있는 폴더 경로 (예: ./dataset/office_background)",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        help="결과를 저장할 폴더 경로 (예: ./dataset/save_img)",
    )
    parser.add_argument(
        "--classes",
        type=str,
        nargs="+",
        default=["Camera", "Phone"],
        help="클래스 목록 (예: Camera Phone)",
    )
    parser.add_argument(
        "--skip-count",
        type=int,
        default=0,
        help="처음 몇 개의 배경 이미지를 건너뛸지 (기본값: 0)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="배경 이미지당 사용할 contour 이미지 개수 (기본값: 5)",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="GUI 다이얼로그를 사용하지 않고 명령줄 인자만 사용",
    )

    args = parser.parse_args()

    # GUI 모드 (명령줄 인자가 없을 때)
    if not args.no_gui and (
        not args.contour_folder or not args.background_folder or not args.output_folder
    ):
        root = tk.Tk()
        root.withdraw()  # 메인 윈도우 숨기기

        print("=" * 60)
        print("Synthetic Data Generator - 폴더 선택")
        print("=" * 60)

        # Contour 이미지 폴더 선택
        if not args.contour_folder:
            print("\n[1/3] Contour 이미지 폴더를 선택하세요...")
            args.contour_folder = filedialog.askdirectory(
                title="Contour 이미지 폴더 선택"
            )
            if not args.contour_folder:
                messagebox.showerror("오류", "Contour 이미지 폴더를 선택해야 합니다.")
                sys.exit(1)
            print(f"✓ 선택된 폴더: {args.contour_folder}")

        # 배경 이미지 폴더 선택
        if not args.background_folder:
            print("\n[2/3] 배경 이미지 폴더를 선택하세요...")
            args.background_folder = filedialog.askdirectory(
                title="배경 이미지 폴더 선택"
            )
            if not args.background_folder:
                messagebox.showerror("오류", "배경 이미지 폴더를 선택해야 합니다.")
                sys.exit(1)
            print(f"✓ 선택된 폴더: {args.background_folder}")

        # 출력 폴더 선택
        if not args.output_folder:
            print("\n[3/3] 결과 저장 폴더를 선택하세요...")
            args.output_folder = filedialog.askdirectory(title="결과 저장 폴더 선택")
            if not args.output_folder:
                messagebox.showerror("오류", "결과 저장 폴더를 선택해야 합니다.")
                sys.exit(1)
            print(f"✓ 선택된 폴더: {args.output_folder}")

        root.destroy()

    # 경로 정규화
    args.contour_folder = os.path.normpath(args.contour_folder)
    args.background_folder = os.path.normpath(args.background_folder)
    args.output_folder = os.path.normpath(args.output_folder)

    # 폴더 존재 확인
    if not os.path.exists(args.contour_folder):
        print(f"오류: Contour 이미지 폴더가 존재하지 않습니다: {args.contour_folder}")
        sys.exit(1)
    if not os.path.exists(args.background_folder):
        print(f"오류: 배경 이미지 폴더가 존재하지 않습니다: {args.background_folder}")
        sys.exit(1)

    # 출력 폴더 생성
    os.makedirs(args.output_folder, exist_ok=True)
    for split in ["train", "valid", "test"]:
        os.makedirs(os.path.join(args.output_folder, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(args.output_folder, split, "labels"), exist_ok=True)

    return args


def main():
    """메인 함수"""
    # 사용자 입력 받기
    args = get_user_inputs()

    class_list = args.classes
    split_list = ["train", "valid", "test"]

    print("\n" + "=" * 60)
    print("설정 정보")
    print("=" * 60)
    print(f"Contour 이미지 폴더: {args.contour_folder}")
    print(f"배경 이미지 폴더: {args.background_folder}")
    print(f"결과 저장 폴더: {args.output_folder}")
    print(f"클래스 목록: {class_list}")
    print(f"배치 크기: {args.batch_size}")
    print(f"건너뛸 배경 이미지 수: {args.skip_count}")
    print("=" * 60 + "\n")

    # Contour 이미지 목록 가져오기
    random.seed("1234")
    cropped_images_list = glob.glob(
        os.path.join(args.contour_folder, "**", "*.*"), recursive=True
    )
    # 이미지 파일만 필터링
    image_extensions = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")
    cropped_images_list = [
        f for f in cropped_images_list if f.lower().endswith(image_extensions)
    ]
    random.shuffle(cropped_images_list)

    if not cropped_images_list:
        print(f"오류: Contour 이미지를 찾을 수 없습니다: {args.contour_folder}")
        sys.exit(1)

    # 배경 이미지 목록 가져오기
    background_list = glob.glob(os.path.join(args.background_folder, "*.*"))
    background_list = [
        f for f in background_list if f.lower().endswith(image_extensions)
    ]

    if not background_list:
        print(f"오류: 배경 이미지를 찾을 수 없습니다: {args.background_folder}")
        sys.exit(1)

    # Contour 이미지를 배치 크기로 재구성
    total_contours = len(cropped_images_list)
    num_batches = (total_contours + args.batch_size - 1) // args.batch_size
    reshaped_contour_image = []
    for i in range(num_batches):
        batch = cropped_images_list[i * args.batch_size : (i + 1) * args.batch_size]
        # 배치 크기만큼 채우기 (부족하면 반복)
        while len(batch) < args.batch_size:
            batch.extend(cropped_images_list[: args.batch_size - len(batch)])
        reshaped_contour_image.append(batch[: args.batch_size])

    print(f"Contour 이미지: {total_contours}개")
    print(f"배경 이미지: {len(background_list)}개")
    print(f"배치 수: {len(reshaped_contour_image)}개\n")

    # 배경 이미지 크기 설정
    bg_area = 1280 * 720
    min_area = int(0.04 * bg_area)
    max_area = int(0.13 * bg_area)

    # 배경 이미지 처리
    for idx2, bg_img in enumerate(background_list):
        if idx2 < args.skip_count:
            continue
        print(f"배경 이미지 처리 중... [{idx2 + 1}/{len(background_list)}]")
        print(f"  파일: {os.path.basename(bg_img)}")

        # 배경 이미지 로드
        try:
            background_img = np.array(Image.open(bg_img))
        except Exception as e:
            print(f"  경고: 배경 이미지 로드 실패 ({bg_img}): {e}")
            continue

        # 배경 이미지 크기 확인 및 조정
        if background_img.shape[0] != 720 or background_img.shape[1] != 1280:
            print(
                f"  정보: 배경 이미지 크기 조정 중 ({background_img.shape[1]}x{background_img.shape[0]} -> 1280x720)"
            )
            background_img = cv2.resize(background_img, (1280, 720))
            background_img = np.array(background_img)

        draw_guideline(
            background_img, background_img.shape[0], background_img.shape[1], 10, 10
        )

        # 현재 배경에 사용할 contour 이미지 배치 선택
        contour_batch = reshaped_contour_image[idx2 % len(reshaped_contour_image)]

        # 각 contour 이미지 처리
        for idx, cropped_image_path in enumerate(
            tqdm(contour_batch, desc="  Contour 처리")
        ):
            img_result = background_img[:, :, :3].copy()

            # Contour 이미지 읽기
            try:
                imgc_alpha = cv2.imread(cropped_image_path, cv2.IMREAD_UNCHANGED)
                if imgc_alpha is None:
                    print(f"    경고: 이미지 로드 실패: {cropped_image_path}")
                    continue
            except Exception as e:
                print(f"    경고: 이미지 로드 오류 ({cropped_image_path}): {e}")
                continue

            # 회전 (현재는 0도)
            angle = 0
            imgc_alpha = rotate_image(imgc_alpha, angle)
            contour_np, cropped_image = find_contour(imgc_alpha)

            # 목표 영역 크기 계산
            target_area = random.randint(min_area, max_area)
            aspect_ratio = cropped_image.shape[1] / cropped_image.shape[0]
            desired_width = int(np.sqrt(target_area * aspect_ratio))
            desired_height = int(
                cropped_image.shape[0] * (desired_width / cropped_image.shape[1])
            )

            # Contour 이미지 리사이즈
            resized_cropped_image = cv2.resize(
                cropped_image, (desired_width, desired_height)
            )
            contour_np, cropped_image = find_contour(resized_cropped_image)

            # 배경 이미지에 배치할 위치 계산
            try:
                rand_width_list = [
                    i
                    for i in range(
                        0,
                        int(background_img.shape[1] - contour_np.shape[1]),
                        max(
                            1, int(background_img.shape[1] - contour_np.shape[1]) // 20
                        ),
                    )
                ]
                rand_height_list = [
                    i
                    for i in range(
                        0,
                        int(background_img.shape[0] - contour_np.shape[0]),
                        max(
                            1, int(background_img.shape[0] - contour_np.shape[0]) // 10
                        ),
                    )
                ]
            except Exception as e:
                print(f"    경고: 위치 계산 오류: {e}")
                continue

            if not rand_width_list or not rand_height_list:
                print(f"    경고: 배치할 공간이 부족합니다: {cropped_image_path}")
                continue

            x = random.sample(rand_width_list, 1)[0]
            y = random.sample(rand_height_list, 1)[0]

            # 클래스 이름 추출 및 바운딩 박스 생성
            name = os.path.basename(cropped_image_path).split("_")[0]
            try:
                class_idx = class_list.index(name)
            except ValueError:
                print(
                    f"    경고: 알 수 없는 클래스 '{name}', 건너뜀: {cropped_image_path}"
                )
                continue

            bounding_box_info, check = create_bbox(
                background_img,
                x,
                y,
                cropped_image.shape[1],
                cropped_image.shape[0],
                class_idx,
            )

            if check == 0:
                print(f"    경고: 바운딩 박스 크기가 1을 초과합니다")
                print(f"      파일: {cropped_image_path}")
                print(
                    f"      크기: {bounding_box_info['width']}, {bounding_box_info['height']}"
                )

            # Alpha 마스크 계산 및 합성
            my_alpha = contour_np[:, :] / 255.0
            cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
            overlay_image_alpha(img_result, cropped_image[:, :, :3], x, y, my_alpha)

            # 데이터셋 분할 (train/valid/test)
            if idx2 < int(len(background_list) * 0.7):
                split_name = "train"
            elif idx2 < int(len(background_list) * 0.85):
                split_name = "valid"
            else:
                split_name = "test"

            # 결과 저장
            save_syn_path = os.path.join(
                args.output_folder, split_name, "images", f"{idx2}_{idx}.jpg"
            )
            save_label_path = os.path.join(
                args.output_folder, split_name, "labels", f"{idx2}_{idx}.txt"
            )

            # 라벨 파일 저장
            with open(save_label_path, "w") as txt_file:
                txt_file.write(
                    f"{bounding_box_info['class']} {bounding_box_info['x_center']} {bounding_box_info['y_center']} {bounding_box_info['width']} {bounding_box_info['height']}\n"
                )

            # 합성 이미지 저장
            Image.fromarray(img_result).save(save_syn_path)

    print("\n" + "=" * 60)
    print("완료!")
    print("=" * 60)
    print(f"결과가 저장되었습니다: {args.output_folder}")


if __name__ == "__main__":
    main()
