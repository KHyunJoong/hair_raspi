import threading
import time
import cv2
import tflite_runtime.interpreter as tflite
import numpy as np
import os
import csv
from collections import deque
import matplotlib.pyplot as plt


# ✅ 첫 번째 모델 (HL Condition 분석)
class_names_condition = ['normal', 'mild', 'moderate', 'severe']

# ✅ 두 번째 모델 (Hair OBD 감지)
class_names_hair = ['1hair', '2hair', '3hair', '4hair']

# ✅ 세 번째 모델 (UNet 기반 Segmentation)
TFLITE_MODEL_PATH = "./app/model/unet_model_200_dynamic_quantized_float32.tflite"

# ✅ UNet 모델 로드 (TFLite)
unet_interpreter = tflite.Interpreter(model_path=TFLITE_MODEL_PATH)
unet_interpreter.allocate_tensors()
unet_input_details = unet_interpreter.get_input_details()
unet_output_details = unet_interpreter.get_output_details()

# ✅ 입력 이미지 전처리 함수
def preprocess_image(image, image_size=512):
    image = cv2.resize(image, (image_size, image_size))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0  # 정규화
    image = np.expand_dims(image, axis=0)
    return image

# ✅ HL Condition 분석 스레드
def hl_condition_script(img_q, result_q):
    model_path = './app/model/hlcM_torchQ16.tflite'
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    while True:
        if not img_q:
            continue
        frame = img_q.popleft()
        image = cv2.resize(frame, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.expand_dims(image, axis=0).astype(np.float32)

        # 추론
        interpreter.set_tensor(input_details[0]['index'], image)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        pred_ind = np.argmax(output_data[0])

        result_q.append(f'HL Condition Predicted: {class_names_condition[pred_ind]}')

# ✅ Hair OBD 감지 스레드
def hair_obd_script(img_q, result_q):
    model_path = './app/model/best_yolo_float32.tflite'
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    image_height = input_details[0]['shape'][1]
    image_width = input_details[0]['shape'][2]

    while True:
        if not img_q:
            continue
        image = img_q.popleft()
        image = cv2.resize(image, (image_width, image_height))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_np = np.array(image) / 255.0
        image_np = np.expand_dims(image_np, axis=0).astype(np.float32)

        interpreter.set_tensor(input_details[0]['index'], image_np)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])[0].T

        boxes_xywh = output[:, :4]
        scores = np.max(output[:, 4:], axis=1)
        classes = np.argmax(output[:, 4:], axis=1)

        threshold = 0.2
        result = []
        print(scores)
        for i, score in enumerate(scores):
            if score > threshold:
                result.append({'class': class_names_hair[classes[i]], 'score': score})

        result_q.append(result)

# ✅ UNet 세그멘테이션 스레드
def unet_segmentation_script(img_q, result_q, threshold=0.65):
    while True:
        if not img_q:
            continue
        original_frame = img_q.popleft()
        image_processed = preprocess_image(original_frame)

        # 모델 입력 설정
        unet_interpreter.set_tensor(unet_input_details[0]['index'], image_processed)

        # 추론 실행
        unet_interpreter.invoke()

        # 결과 가져오기
        output_data = unet_interpreter.get_tensor(unet_output_details[0]['index'])
        output_mask = (output_data[0, :, :, 0] > threshold).astype(np.uint8)  # Threshold 적용

        result_q.append((image_processed, output_mask))

# ✅ 메인 실행 함수
def run_scripts():
    img_q_condition = deque(maxlen=3)
    result_q_condition = deque(maxlen=10)

    img_q_hair = deque(maxlen=3)
    result_q_hair = deque(maxlen=10)

    img_q_unet = deque(maxlen=3)
    result_q_unet = deque(maxlen=10)

    # ✅ 스레드 시작
    thread_a = threading.Thread(target=hl_condition_script, args=(img_q_condition, result_q_condition))
    thread_b = threading.Thread(target=hair_obd_script, args=(img_q_hair, result_q_hair))
    thread_c = threading.Thread(target=unet_segmentation_script, args=(img_q_unet, result_q_unet))

    thread_a.start()
    thread_b.start()
    thread_c.start()

    # ✅ 카메라 캡처 시작
    cap = cv2.VideoCapture(1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ✅ 프레임 큐에 추가
        img_q_condition.append(frame)
        img_q_hair.append(frame)
        img_q_unet.append(frame)

        # ✅ HL Condition 결과 출력
        if result_q_condition:
            condition_result = result_q_condition.popleft()
            print(condition_result)

        # ✅ Hair OBD 결과 출력
        if result_q_hair:
            hair_result = result_q_hair.popleft()
            print(f'Detected Hair: {hair_result}')

        # ✅ UNet 세그멘테이션 결과 표시
        if result_q_unet:
            original_frame, mask = result_q_unet.popleft()
            unique, counts = np.unique(mask, return_counts=True)
            for val, count in zip(unique, counts):
                print(f"값 {val}: {count}개")
            # ✅ 마스크를 원본 프레임에 오버레이
            print("original_frame shape:", original_frame.shape)
            print("mask shape before resize:", mask.shape)
            print("original_frame dtype:", original_frame.dtype)
            original_frame = (np.squeeze(original_frame) * 255).astype(np.uint8)
            overlay = cv2.applyColorMap(mask * 255, cv2.COLORMAP_JET)
            print("original_frame dtype:", original_frame.dtype)
            print("overlay dtype:", overlay.dtype)
            print("original_frame shape:", original_frame.shape)
            print("mask overlay shape before resize:", overlay.shape)
            blended = cv2.addWeighted(original_frame, 0.7, overlay, 0.3, 0)
            cv2.imshow('UNet Segmentation', blended)
        # ✅ 원본 프레임 표시
        cv2.imshow('HL Condition', frame)

        # ✅ 종료 조건
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # ✅ 리소스 정리
    cap.release()
    thread_a.join()
    thread_b.join()
    thread_c.join()
    cv2.destroyAllWindows()

# ✅ 메인 실행
if __name__ == "__main__":
    run_scripts()

