import os
import cv2
from fastapi import APIRouter, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import StreamingResponse, FileResponse
from services.services import analysis
# 템플릿 및 이미지 경로 설정
template_path = "templates"
image_path = "static/captured.jpg"
processed_path = "static/processed.jpg"

# APIRouter 생성
router = APIRouter()
templates = Jinja2Templates(directory=template_path)

# 카메라 스트리밍 설정
cap = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@router.get("/", response_class=FileResponse)
def stream_page(request: Request):
    return templates.TemplateResponse("stream.html", {"request": request})

@router.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@router.get("/capture")
def capture_image():
    success, frame = cap.read()
    if success:
        if os.path.exists(image_path):
            os.remove(image_path)
        cv2.imwrite(image_path, frame)
    return {"message": "Image captured"}

@router.get("/view", response_class=FileResponse)
def view_page(request: Request):
    return templates.TemplateResponse("view.html", {"request": request})

@router.get("/analyze")
def analyze_image():
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")
    # a,b1,b2,b3,c = analysis(cv2.imread("/static/captured.jpg"))
    # print("a",a,"b",b1,b2,b3,"c",c)
    img = cv2.imread(image_path)
    condition_result, hair_result1,hair_result2,hair_result3, analysis_image = analysis(img)
    print(hair_result2,hair_result3)
    cv2.imwrite(processed_path, analysis_image)
    return {"message": f"Image processed{condition_result},{ hair_result2},{hair_result3}"}

@router.get("/result", response_class=FileResponse)
def result_page(request: Request):
    return templates.TemplateResponse("result.html", {"request": request})

@router.get("/get_image")
def get_image():
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(image_path, media_type="image/jpeg")

@router.get("/get_processed")
def get_processed():
    if not os.path.exists(processed_path):
        raise HTTPException(status_code=404, detail="Processed image not found")
    return FileResponse(processed_path)
