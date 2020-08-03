import os
import sys
import cv2
import threading
import numpy as np

#%% 프로젝트 폴더를 sys.path에 추가(Jetson Nano에서 직접 실행할 때 필요)
project_path = "/home/jetson/MyWorkspace/AI-ROVER/detection"
sys.path.append(project_path)

from utils.trt_ssd_object_detect import TrtThread, BBoxVisualization
from utils.coco_label_map import CLASSES_DICT
import time

def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=60,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

def canny(img, kernel_size, low_threshold, high_threshold):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    img = cv2.Canny(img, low_threshold, high_threshold)
    return img

def roi(img, vertices, color3=(255, 255, 255), color1=255):
    mask = np.zeros_like(img)  # mask=img와 같은 크기의 빈 이미지

    if len(img.shape) > 2:  # 3채널 컬러 이미지라면
        color = color3
    else:  # 1채널 흑백 이미지라면
        color = color1

    # 정점들로 이루어진 다각형(ROI부분)을 color로 채움
    cv2.fillPoly(mask, vertices, color)

    # 이미지와 color로 채워진 ROI 합치기
    ROI_image = cv2.bitwise_and(img, mask)
    return ROI_image

def draw_lines(img, lines, color=[0, 0, 255], thickness=2):  # 선 그리기
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):  # 허프 변환
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)

    return lines

def weighted_img(img, initial_img, α=1, β=1., λ=0.):  # 두 이미지 operlap 하기
    return cv2.addWeighted(initial_img, α, img, β, λ)

#%% 감지 결과 활용(처리)
def handleDetectedObject(trtThread, condition):
    # 전체 스크린 플래그 변수
    full_scrn = False

    # 초당 프레임 수
    fps = 0.0

    # 시작 시간
    tic = time.time()

    # 바운딩 박스 시각화 객체
    vis = BBoxVisualization(CLASSES_DICT)

    # TrtThread가 실행 중일 때 반복 실행
    while trtThread.running:
        with condition:
            # 감지 결과가 있을 때까지 대기
            condition.wait()
            # 감지 결과 얻기
            img, boxes, confs, clss = trtThread.getDetectResult()

        canny_img = canny(img, 3, 70, 210)
        height, width = img.shape[:2]
        
        vertices = np.array(
             [[(50, height), (width / 2 - 45, height / 2 + 60), (width / 2 + 45, height / 2 + 60),
               (width - 50, height)]],
             dtype=np.int32)
        
        ROI_img = roi(canny_img, vertices)  # ROI 설정
        
        line_arr = hough_lines(ROI_img, 1, 1 * np.pi / 180, 30, 10, 20)  # 허프 변환
        line_arr = np.squeeze(line_arr)
        
         # 기울기 구하기
        slope_degree = (np.arctan2(line_arr[:, 1] - line_arr[:, 3], line_arr[:, 0] - line_arr[:, 2]) * 180) / np.pi
        
         # 수평 기울기 제한
        line_arr = line_arr[np.abs(slope_degree) < 160]
        slope_degree = slope_degree[np.abs(slope_degree) < 160]
         # 수직 기울기 제한
        line_arr = line_arr[np.abs(slope_degree) > 95]
        slope_degree = slope_degree[np.abs(slope_degree) > 95]
         # 필터링된 직선 버리기
        L_lines, R_lines = line_arr[(slope_degree > 0), :], line_arr[(slope_degree < 0), :]
        temp = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        L_lines, R_lines = L_lines[:, None], R_lines[:, None]
         # 직선 그리기
        draw_lines(temp, L_lines)
        draw_lines(temp, R_lines)
        
        result = weighted_img(temp, img)  # 원본 이미지에 검출된 선 overlap

        ##############################

        # 감지 결과 출력
        img = vis.drawBboxes(img, boxes, confs, clss)

        # 초당 프레임 수 드로잉
        img = vis.drawFps(img, fps)

        #fin = weighted_img(result, img)

        # 이미지를 윈도우에 보여주기
        cv2.imshow("obj_detect_from_video", img)
        cv2.imshow("line_detect_from_video", result)
        # cv2.imshow("detect_from_video", fin)

        # 초당 프레임 수 계산
        toc = time.time()
        curr_fps = 1.0 / (toc-tic)
        fps = curr_fps if fps == 0.0 else (fps * 0.95 + curr_fps * 0.05)    # 지수 감소 평균
        tic = toc

        # 키보드 입력을 위해 1ms 동안 대기, 입력이 없으면 -1을 리턴
        key = cv2.waitKey(1)
        if key == 27:
            # ESC를 눌렀을 때
            break
        elif key == ord("F") or key == ord("f"):
            # F나 f를 눌렀을 경우 전체 스크린과 토글 기능
            full_scrn = not full_scrn
            if full_scrn:
                cv2.setWindowProperty("detect_from_video", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty("detect_from_video", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

#%% 메인 함수
def main():
    # 엔진 파일 경로
    enginePath = project_path + "/models/ssd_mobilenet_v1_coco_2018_01_28/tensorrt_fp16.engine"
    # 비디오 캡처 객체 얻기
    videoCapture = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    # videoCapture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    # videoCapture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    # 감지 결과(생산)와 처리(소비)를 동기화를 위한 Condition 얻기
    condition = threading.Condition()
    # TrtThread 객체 생성
    trtThread = TrtThread(enginePath, TrtThread.INPUT_TYPE_USBCAM, videoCapture, 0.3, condition)
    # 감지 시작
    trtThread.start()

    # 이름 있는 윈도우 만들기
    cv2.namedWindow("detect_from_video", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("detect_from_video", 320, 240)
    cv2.setWindowTitle("detect_from_video", "detect_from_video")

    # 감지 결과 처리(활용)
    handleDetectedObject(trtThread, condition)

    # 감지 중지
    trtThread.stop()

    # VideoCapture 중지
    videoCapture.release()

    # 윈도우 닫기
    cv2.destroyAllWindows()

def image_main():
    image = cv2.imread('track.jpg')
    height, width = image.shape[:2]

    canny_img = canny(image, 3, 70, 210)
    vertices = np.array(
        [[(50, height), (width / 2 - 45, height / 2 + 60), (width / 2 + 45, height / 2 + 60), (width - 50, height)]],
        dtype=np.int32)
    ROI_img = roi(canny_img, vertices)  # ROI 설정

    line_arr = hough_lines(ROI_img, 1, 1 * np.pi / 180, 30, 10, 20)  # 허프 변환
    line_arr = np.squeeze(line_arr)

    # 기울기 구하기
    slope_degree = (np.arctan2(line_arr[:, 1] - line_arr[:, 3], line_arr[:, 0] - line_arr[:, 2]) * 180) / np.pi

    # 수평 기울기 제한
    line_arr = line_arr[np.abs(slope_degree) < 160]
    slope_degree = slope_degree[np.abs(slope_degree) < 160]
    # 수직 기울기 제한
    line_arr = line_arr[np.abs(slope_degree) > 95]
    slope_degree = slope_degree[np.abs(slope_degree) > 95]
    # 필터링된 직선 버리기
    L_lines, R_lines = line_arr[(slope_degree > 0), :], line_arr[(slope_degree < 0), :]
    temp = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    L_lines, R_lines = L_lines[:, None], R_lines[:, None]
    # 직선 그리기
    draw_lines(temp, L_lines)
    draw_lines(temp, R_lines)

    result = weighted_img(temp, image)  # 원본 이미지에 검출된 선 overlap
    cv2.imshow('result', result)  # 결과 이미지 출력
    cv2.waitKey(0)

#%% 최상위 스크립트 실행
if __name__ == "__main__":
    main()
