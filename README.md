# SeSac AI 두피 탈모 관리 서비스 시스템

## 📌 프로젝트 개요
사용자의 두피 상태를 정밀 측정하고, 실시간으로 탈모를 관리하는 AI 기반 시스템입니다. 확대 현미경으로 촬영한 두피 이미지를 분석하여 모공 개수, 모공당 모발 개수, 모발 면적 등을 측정하고 상태를 분류하여 시각적으로 제공합니다.

---

## 👥 팀 구성
- **총원**: 3명
  - 개발 1명
  - 기획 2명

---

## 🛠️ 주요 기술 스택
- **Object Detection**: YOLOv11n
- **Deep Learning Frameworks**: PyTorch, TensorFlow Lite
- **Segmentation**: U-Net
- **Web/API**: FastAPI
- **Edge Computing**: Raspberry Pi 5
- **Frontend**: HTML

---

## 📌 주요 기능 및 업무
- **이미지 촬영 및 처리**
  - 확대 현미경으로 두피 촬영
  - OpenCV를 이용한 이미지 처리

- **AI 모델 개발 및 경량화**
  - YOLOv11n 기반의 객체 탐지로 모공당 모발 개수 검출
  - U-Net 모델을 활용한 이미지 분할(Segmentation)로 모발 영역 정량 분석
  - TensorFlow Lite를 사용한 모델 경량화 (엣지 컴퓨팅 최적화)

- **API 서버 및 웹 서비스 구축**
  - FastAPI로 RESTful API 서버 구축
  - HTML 렌더링을 통한 사용자 인터페이스 제공

---

## 📂 시스템 구조도
### 시스템 아키텍처
![시스템 아키텍처](![systemArchitecture](https://github.com/user-attachments/assets/bbcbcff1-94b8-410c-a0b8-f90071e85227)
)

### 액티비티 아키텍처
![액티비티 아키텍처](![activity diagram](https://github.com/user-attachments/assets/df74dfc0-1c54-492b-b4be-af874807f625)
)

---

## 📂 AI 모델 구성
### 모공당 모발 개수 (YOLOv11)
![YOLO 모공당 모발 개수]()

### 두피 상태 분류 (EfficientNet, MobileNet 기반 CNN)
![두피 상태 분류](![CNN](https://github.com/user-attachments/assets/6851773f-5ece-4886-a38d-09f35415f627)
)

### 두피 면적 대비 모발 분포 (U-Net Segmentation)
![두피 면적 대비 모발 분포]()

---

## 📱 서비스 UI 디자인
- 날짜 기록, 상세 리포트, 변화 추이 화면 제공:
- (![그림7](https://github.com/user-attachments/assets/ef95bff5-94cf-4db1-93e1-ab2c6dd6eb1b)
)

---
"""

# Save to a markdown file
readme_path = Path("/mnt/data/README.md")
readme_path.write_text(readme_content, encoding="utf-8")

readme_path.name
