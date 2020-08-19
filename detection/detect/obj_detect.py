import os
import sys
import cv2
import threading
import base64
import datetime
import paho.mqtt.client as mqtt

# 프로젝트 폴더를 sys.path에 추가(Jetson Nano에서 직접 실행할 때 필요)
project_path = "/home/jetson/MyWorkspace/Ambulance/detection"
current_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(current_path)

from ambulance.ambulance import Ambulance
from detection.utils.trt_ssd_object_detect import TrtThread, BBoxVisualization
from detection.utils.sign_label_map import CLASSES_DICT
import time

class ResultImageMqttClient:
    def __init__(self, brokerIp=None, brokerPort=1883, pubTopic=None, ambulance=None, camera=None):
        self.brokerIp = brokerIp
        self.brokerPort = brokerPort
        self.pubTopic = pubTopic
        self.client = mqtt.Client()
        self.client.on_connect = self.__on_connect
        self.client.on_disconnect = self.__on_disconnect
        self.client.on_message = self.__on_message
        self.ambulance = ambulance
        self.camera = camera
        self.camera.camera_init()

        self.prev_position = None

        self.trtThread = None

        # 횡단보도, 정지사인, 신호등(RED) 정지 플래그
        self.stop_flag = False


        # 동기화
        self.__netflag = False
        self.set_sub = set()
        self.set_res = set()
        self.tic = datetime.datetime.now()
        self.toc = datetime.datetime.now()

    def start(self):
        self.client.connect(self.brokerIp, self.brokerPort)
        self.client.loop_start()
        self.main(self.camera, self.ambulance)
        self.client.loop_stop()

    def __on_connect(self, client, userdata, flags, rc):
        print("ResultImageMqttClient mqtt broker connect")
        self.client.subscribe("command/1/process/stop")
        self.client.subscribe("/res/ambulance1")

    def __on_disconnect(self, client, userdata, rc):
        print("ResultImageMqttClient mqtt broker disconnect")

    def __on_message(self, client, userdata, message):
        if message.topic == "command/1/process/stop":
            self.trtThread.stop()
            self.disconnect()
        elif message.topic == "/res/ambulance1":
            content = str(message.payload, encoding="utf-8")
            self.set_res.add(content)
            self.set_sub.add(content)

            if self.set_res == self.set_sub:
                self.tic = datetime.datetime.now()
                self.__netflag = True

            else:
                self.toc = datetime.datetime.now()
                if self.toc - self.tic > datetime.timedelta(seconds=2):
                    self.set_sub = self.set_sub.intersection(self.set_res)
                    print(self.set_sub)

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
        if self.__netflag:
            # MQTT Broker로 보내기
            self.client.publish(self.pubTopic, b64_bytes)
            self.__netflag = False
            self.set_res.clear()
        # print("send!")

    # 감지 결과 활용(처리)
    def handleDetectedObject(self, trtThread, condition):
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
            # 감지된 객체 인덱스
            for cls, box in zip(clss, boxes):
                # 알파벳 인덱스 범위에 포함되면, 위치 정보 저장 및 발행
                if cls in range(13, 28, 1):
                    # 이전의 위치정보와 비교해서 다를 때만 저장 및 발행
                    if not (self.prev_position == CLASSES_DICT.get(cls)):
                        self.ambulance.set_position(CLASSES_DICT.get(cls))
                        self.client.publish("ambulance/1/position", self.ambulance.get_position())
                        print(CLASSES_DICT.get(cls))
                        self.prev_position = self.ambulance.get_position()

                # 장애물(cone)
                if cls == 11:
                    # 바운딩 박스 크기, 위치 조사
                    x1, y1, x2, y2 = box
                    box_area = abs((x2 - x1) * (y2 - y1))
                    box_center_x = int((x2 + x1) / 2)

                    # print("{}, ".format(box_area), end="")
                    # print(box_center_x)

                    # 박스 위치가 현재 차선에 속할 때
                    if img.shape[1] * (1/3) < box_center_x < img.shape[1] * (2/3):
                        # 박스 크기가 충분히 클 때(가까울 때)
                        if box_area > 2000:
                            # 차선변경 플래그가 False일때만 실행 -> 차선변경을 하는 중간에 다시 실행되지 않기 위함
                            if not self.ambulance.change_road_flag:
                                # 자율 주행 모드인 경우에만 차선 변경
                                if self.ambulance.get_mode() == Ambulance.AUTO_MODE:
                                    self.ambulance.change_road()

                # 횡단보도(crosswalk) 아직 안됨
                if cls == 4:
                    if self.ambulance.get_mode() == Ambulance.AUTO_MODE and not self.ambulance.stop_flag:
                        self.ambulance.set_mode(Ambulance.MANUAL_MODE)
                        self.ambulance.stop()
                    else:
                        if self.ambulance.stop_count < 105:
                            if self.ambulance.stop_count > 100:
                                self.ambulance.forward(0.9)
                            elif self.ambulance.stop_count == 104:
                                self.ambulance.set_mode(Ambulance.AUTO_MODE)
                            self.ambulance.stop_count += 1

                # 방지턱(bump)
                if cls == 12:
                    pass

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
        enginePath = project_path + "/models/ssd_mobilenet_v2_sign9/tensorrt_fp16.engine"
        # 비디오 캡처 객체 얻기
        videoCapture = camera
        # 감지 결과(생산)와 처리(소비)를 동기화를 위한 Condition 얻기
        condition = threading.Condition()
        # TrtThread 객체 생성
        self.trtThread = TrtThread(enginePath, TrtThread.INPUT_TYPE_USBCAM, videoCapture, 0.6, condition, ambulance)
        # 감지 시작
        self.trtThread.start()

        # 감지 결과 처리(활용)
        self.handleDetectedObject(self.trtThread, condition)

        # 감지 중지
        self.trtThread.stop()

        # VideoCapture 중지
        videoCapture.release()
#%% 최상위 스크립트 실행
if __name__ == "__main__":
    car = Ambulance()
    thread = ResultImageMqttClient("192.168.3.132", 1883, "ambulance/camera/frameLine", car)
    thread.start()
