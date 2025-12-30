# Synthetic Data Generator

합성 데이터 생성 도구 - rembg를 사용하여 전경/배경을 분리하고, 객체를 배경 이미지에 합성하여 YOLO 형식의 라벨을 생성합니다.

## 📋 개요

이 프로젝트는 객체 검출 모델 학습을 위한 합성 데이터를 생성하는 도구입니다. rembg를 사용하여 전경과 배경을 분리한 후, 객체를 다양한 배경 이미지에 합성하여 데이터셋을 생성합니다.

## 🎯 주요 기능

1. **전경/배경 분리**: rembg를 사용하여 객체의 마스크 생성
2. **배경 합성**: 객체를 배경 이미지에 자연스럽게 합성
3. **YOLO 라벨 생성**: 합성된 이미지에 대한 YOLO 형식 바운딩 박스 라벨 자동 생성
4. **데이터셋 분할**: train/val/test 자동 분할

## 📁 프로젝트 구조

```
synthetic-data-generator/
├── README.md
├── LICENSE
├── requirements.txt
├── tools/
│   └── contour_tools.py      # 컨투어 찾기, 바운딩 박스 생성 유틸리티
├── scripts/
│   ├── synthetic_generator.py    # 메인 합성 스크립트
│   ├── resize_background.py      # 배경 이미지 리사이즈
│   └── label_converter.py        # 라벨 클래스 변환
└── .gitignore
```

## 🔧 설치

### 필수 요구사항

- Python 3.7 이상
- OpenCV (cv2)
- NumPy
- Pillow
- rembg (https://github.com/danielgatis/rembg)

### 설치 방법

```bash
# 저장소 클론
git clone <repository-url>
cd synthetic-data-generator

# 의존성 설치
pip install -r requirements.txt

# rembg 설치
pip install rembg
```

## 📖 사용법

### 1. 합성 데이터 생성

```bash
python scripts/synthetic_generator.py
```

**설정**:
- `root_path`: 데이터셋 루트 경로
- `save_path`: 결과 저장 경로
- `class_list`: 클래스 목록 (예: ['Camera', 'Phone'])

### 2. 배경 이미지 리사이즈

```bash
python scripts/resize_background.py
```

배경 이미지를 일정 크기(1280x720)로 리사이즈합니다.

### 3. 라벨 클래스 변환

```bash
python scripts/label_converter.py
```

YOLO 라벨 파일의 클래스 번호를 변환합니다.

## 📝 데이터 구조

```
dataset/
├── contour_image/          # rembg로 처리된 전경 이미지 (RGBA)
│   ├── Camera/
│   └── Phone/
├── office_background/        # 배경 이미지
└── save_img/               # 생성된 합성 데이터
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── valid/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/
```

## 🔍 프로세스

1. **객체 크롭**: 원본 이미지에서 객체 영역 추출
2. **rembg 처리**: 전경/배경 분리 및 마스크 생성
3. **배경 합성**: 객체를 배경 이미지에 합성
4. **라벨 생성**: YOLO 형식 바운딩 박스 라벨 생성

## 📄 라이선스

[LICENSE 파일 참조](LICENSE)

## 🤝 기여

이슈 리포트 및 Pull Request를 환영합니다!
