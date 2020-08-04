import os
import sys
import cv2
import threading
import base64
import numpy as np
import paho.mqtt.client as mqtt

# 프로젝트 폴더를 sys.path에 추가(Jetson Nano에서 직접 실행할 때 필요)
project_path = "/home/jetson/MyWorkspace/Ambulance/detection"
current_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(current_path)

from ambulance.ambulance import Ambulance
from detection.utils.trt_ssd_object_detect import TrtThread, BBoxVisualization
from detection.utils.coco_label_map import CLASSES_DICT
import time

class ResultImageMqttClient:
    def __init__(self, brokerIp=None, brokerPort=1883, pubTopic=None, ambulance=None, camera=None):
        self.brokerIp = brokerIp
        self.brokerPort = brokerPort
        self.pubTopic = pubTopic
        self.client = mqtt.Client()
        self.client.on_connect = self.__on_connect
        self.client.on_disconnect = self.__on_disconnect
        self.ambulance = ambulance
        self.camera = camera
        self.camera.camera_init()
        print("camera instance constructed")

    def start(self):
        # thread = threading.Thread(target=self.main, args=[self.camera, self.ambulance], daemon=False)
        # thread.start()
        self.client.connect(self.brokerIp, self.brokerPort)
        self.client.loop_start()
        self.main(self.camera, self.ambulance)
        self.client.loop_stop()

    def __on_connect(self, client, userdata, flags, rc):
        print("ResultImageMqttClient mqtt broker connect")

    def __on_disconnect(self, client, userdata, rc):
        print("ResultImageMqttClient mqtt broker disconnect")

    def disconnect(self):
        self.client.disconnect()

    def sendBase64(self, frame):
        if self.client is None:
            return
        # MQTT Broker가 연결되어 있지 않을 경우
        if not self.client.is_connected():
            return
        # JPEG 포맷으로 인코딩
        retval, bytes = cv2.imencode(".jpg", frame)
        # 인코딩이 실패했을 경우
        if not retval:
            print("image encoding fail")
            return
        # Base64 문자열로 인코딩
        b64_bytes = base64.b64encode(bytes)
        # MQTT Broker로 보내기
        self.client.publish(self.pubTopic, b64_bytes)
        print("send!")

    # 감지 결과 활용(처리)
    def handleDetectedObject(self, trtThread, condition):
        # print("handleDetectedObject method")
        # 초당 프레임 수
        fps = 0.0

        # 시작 시간
        tic = time.time()

        # 바운딩 박스 시각화 객체
        vis = BBoxVisualization(CLASSES_DICT)

        # print("before while")
        # TrtThread가 실행 중일 때 반복 실행
        while trtThread.running:
            # print("in while")
            with condition:
                # 감지 결과가 있을 때까지 대기
                condition.wait()
                # 감지 결과 얻기
                img, boxes, confs, clss = trtThread.getDetectResult()

            # 감지 결과 출력
            img = vis.drawBboxes(img, boxes, confs, clss)

            # 초당 프레임 수 드로잉
            img = vis.drawFps(img, fps)

            self.sendBase64(img)

            # 초당 프레임 수 계산
            toc = time.time()
            curr_fps = 1.0 / (toc-tic)
            fps = curr_fps if fps == 0.0 else (fps * 0.95 + curr_fps * 0.05)    # 지수 감소 평균
            tic = toc



    # 메인 함수
    def main(self, camera, ambulance):
        # 엔진 파일 경로
        enginePath = project_path + "/models/ssd_mobilenet_v1_coco_2018_01_28/tensorrt_fp16.engine"
        # 비디오 캡처 객체 얻기
        videoCapture = camera
        # 감지 결과(생산)와 처리(소비)를 동기화를 위한 Condition 얻기
        condition = threading.Condition()
        # TrtThread 객체 생성
        trtThread = TrtThread(enginePath, TrtThread.INPUT_TYPE_USBCAM, videoCapture, 0.3, condition, ambulance)
        # 감지 시작
        trtThread.start()

        # 감지 결과 처리(활용)
        self.handleDetectedObject(trtThread, condition)

        # 감지 중지
        trtThread.stop()

        # VideoCapture 중지
        videoCapture.release()
#%% 최상위 스크립트 실행
if __name__ == "__main__":
    car = Ambulance()
    thread = ResultImageMqttClient("192.168.3.132", 1883, "ambulance/camera/frameLine", car)
    thread.start()
