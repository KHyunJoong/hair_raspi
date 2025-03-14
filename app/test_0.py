import cv2
import tflite_runtime.interpreter as tflite
import numpy as np

# ✅ UNet 모델 로드 (TFLite)
TFLITE_MODEL_PATH = "./model/unet_model_200_dynamic_quantized_float32.tflite"
unet_interpreter = tflite.Interpreter(model_path=TFLITE_MODEL_PATH)
unet_interpreter.allocate_tensors()
unet_input_details = unet_interpreter.get_input_details()
unet_output_details = unet_interpreter.get_output_details()

def preprocess_image(image, image_size=512):
    image = cv2.resize(image, (image_size, image_size))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0  # 정규화
    image = np.expand_dims(image, axis=0)
    return image

def hl_condition_script(frame):
    """ HL Condition 분석 모델 """
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

    class_names_condition = ['normal', 'mild', 'moderate', 'severe']
    return class_names_condition[pred_ind]

def hair_obd_script(frame):
    """ Hair OBD 감지 (YOLO 모델) """
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

    boxes_xywh = output[:, :4]  # 바운딩 박스 좌표 (cx, cy, w, h)
    scores = np.max(output[:, 4:], axis=1)
    classes = np.argmax(output[:, 4:], axis=1)

    threshold = 0.2
    result = []
    cropped_images = []
    class_names_hair = ['1hair', '2hair', '3hair', '4hair']
    pores = 0
    hairs = 0
    detected_boxes = []

    for i, score in enumerate(scores):
        if score > threshold:
            x, y, w, h = boxes_xywh[i]
            x, y, w, h = int(x), int(y), int(w), int(h)

            # 유효한 영역인지 확인
            if x < 0 or y < 0 or w <= 0 or h <= 0:
                print(f"⚠️ Invalid bounding box detected: (x={x}, y={y}, w={w}, h={h})")
                continue  # 무효한 박스는 스킵

            cropped_img = frame[y:y+h, x:x+w]

            if cropped_img is None or cropped_img.size == 0:
                print(f"⚠️ Skipping empty cropped image at (x={x}, y={y}, w={w}, h={h})")
                continue

            cropped_images.append((cropped_img, (x, y, w, h)))  # 좌표 함께 저장
            result.append({'class': class_names_hair[classes[i]], 'score': score})
            pores += 1
            hairs += (i + 1)

            # 감지된 박스 저장 (시각화에 사용)
            detected_boxes.append((x, y, w, h, class_names_hair[classes[i]], score))

    return result, pores, hairs, cropped_images, detected_boxes

def unet_segmentation_script(frame):
    original_frame = frame
    image_processed = preprocess_image(original_frame)

    # 모델 입력 설정
    unet_interpreter.set_tensor(unet_input_details[0]['index'], image_processed)

    # 추론 실행
    unet_interpreter.invoke()

    # 결과 가져오기
    output_data = unet_interpreter.get_tensor(unet_output_details[0]['index'])
    output_mask = (output_data[0, :, :, 0] > 0.65).astype(np.uint8)  # Threshold 적용

    return output_mask

def analysis(frame):
    if frame is None:
        print("❌ Error: Input frame is empty. Check image path and validity.")
        return None

    condition_result = hl_condition_script(frame)
    hair_result1, hair_result2, hair_result3, cropped_images, detected_boxes = hair_obd_script(frame)

    # UNet 결과를 저장할 마스크 (원본 크기)
    full_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

    print(f"📌 Detected {len(cropped_images)} cropped images for segmentation")

    for idx, (cropped_img, (x, y, w, h)) in enumerate(cropped_images):
        if cropped_img is None or cropped_img.size == 0:
            print(f"⚠️ Skipping empty cropped image at (x={x}, y={y}, w={w}, h={h})")
            continue

        mask = unet_segmentation_script(cropped_img)  # UNet 실행

        # 크롭된 영역을 원본 이미지 크기에 매핑
        resized_mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        full_mask[y:y+h, x:x+w] = resized_mask

    # 원본 이미지와 UNet 마스크 시각화
    overlay = cv2.applyColorMap((full_mask * 255).astype(np.uint8), cv2.COLORMAP_JET)
    blended = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

    # ✅ 감지된 영역에 박스 및 클래스명 추가
    for (x, y, w, h, class_name, score) in detected_boxes:
        cv2.rectangle(blended, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 초록색 박스
        text = f"{class_name} ({score:.2f})"
        cv2.putText(blended, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return condition_result, hair_result1, hair_result2, hair_result3, blended

# 실행
a, b1, b2, b3, c = analysis(cv2.imread("./static/captured.jpg"))

# ✅ UNet 결과 시각화 (박스 포함)
cv2.imshow('UNet Segmentation + Detected Boxes', c)
cv2.waitKey(0)
cv2.destroyAllWindows()
