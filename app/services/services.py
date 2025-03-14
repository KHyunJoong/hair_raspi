import threading
import time
import cv2
import tflite_runtime.interpreter as tflite
import numpy as np
import os
import csv


# ✅ 두 번째 모델 (Hair OBD 감지)


# ✅ 세 번째 모델 (UNet 기반 Segmentation)
TFLITE_MODEL_PATH = "./model/unet_model_200_dynamic_quantized_float32.tflite"

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
def hl_condition_script(frame):
    model_path = './model/hlcM_torchQ16.tflite'
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    image = cv2.resize(frame, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.expand_dims(image, axis=0).astype(np.float32)

    # 추론
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    pred_ind = np.argmax(output_data[0])
    # ✅ 첫 번째 모델 (HL Condition 분석)
    class_names_condition = ['normal', 'mild', 'moderate', 'severe']
    # result_q.append(f'HL Condition Predicted: {class_names_condition[pred_ind]}')
    return class_names_condition[pred_ind]
# ✅ Hair OBD 감지 스레드
def hair_obd_script(frame):
    model_path = './model/best_yolo_float32.tflite'
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    image_height = input_details[0]['shape'][1]
    image_width = input_details[0]['shape'][2]

    image = cv2.resize(frame, (image_width, image_height))
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
    class_names_hair = ['1hair', '2hair', '3hair', '4hair']
    pores = 0
    hairs = 0
    for i, score in enumerate(scores):
        if score > threshold:
            result.append({'class': class_names_hair[classes[i]], 'score': score})
            pores += 1
            hairs += (i + 1)
    # result_q.append(result)
    return result, pores, hairs
# ✅ UNet 세그멘테이션 스레드
def unet_segmentation_script(frame, threshold=0.65):

        original_frame = frame
        image_processed = preprocess_image(original_frame)

        # 모델 입력 설정
        unet_interpreter.set_tensor(unet_input_details[0]['index'], image_processed)

        # 추론 실행
        unet_interpreter.invoke()

        # 결과 가져오기
        output_data = unet_interpreter.get_tensor(unet_output_details[0]['index'])
        output_mask = (output_data[0, :, :, 0] > threshold).astype(np.uint8)  # Threshold 적용

        return image_processed, output_mask

def analysis(frame):
    condition_result = hl_condition_script(frame)
    hair_result1, hair_result2, hair_result3 = hair_obd_script(frame)
    result_q_unet = unet_segmentation_script(frame)

    # ✅ UNet 결과

    original_frame, mask = result_q_unet
    unique, counts = np.unique(mask, return_counts=True)
    for val, count in zip(unique, counts):
        print(f"값 {val}: {count}개")

    # ✅ 원본 프레임 변환
    original_frame = (np.squeeze(original_frame) * 255).astype(np.uint8)

    # ✅ UNet 마스크 변환
    overlay = cv2.applyColorMap((mask * 255).astype(np.uint8), cv2.COLORMAP_JET)
    blended = cv2.addWeighted(original_frame, 0.7, overlay, 0.3, 0)


    return condition_result, hair_result1,hair_result2,hair_result3, blended


