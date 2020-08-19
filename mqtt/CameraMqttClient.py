import cv2
from datetime import datetime
import paho.mqtt.client as mqtt
import threading
import base64
from collections import deque

import numpy as np
import os
import sys

current_path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(current_path)

from utils.camera import Camera
import line_detect.line_detect as line
import utils.pid_controller as pid

class ImageMqttPublisher:
    def __init__(self, brokerIp=None, brokerPort=1883, pubTopic=None, ambulance=None):
        self.brokerIp = brokerIp
        self.brokerPort = brokerPort
        self.pubTopic = pubTopic
        self.client = mqtt.Client()
        self.client.on_connect = self.__on_connect
        self.client.on_disconnect = self.__on_disconnect
        self.ambulance = ambulance
        # self.client.on_message = self.__on_message
        self.camera = Camera(cap_w=320, cap_h=240, dp_w=320, dp_h=240, fps=10, flip_method=0)
        self.camera.camera_init()
        print("camera instance constructed")
        # self.__cnt = 0

    def start(self):
        thread = threading.Thread(target=self.__run, daemon=True)
        thread.start()

    def __run(self):
        self.client.connect(self.brokerIp, self.brokerPort)

        road_half_width_list = deque(maxlen=10)
        road_half_width_list.append(165)
        pid_controller = pid.PIDController(round(datetime.utcnow().timestamp() * 1000))
        pid_controller.set_gain(0.55, -0.001, 0.23)
#        pid_controller.set_gain(0.63, -0.001, 0.23)

        self.client.loop_start()
        while True:
            if self.camera.isOpened():
                retval, frame = self.camera.read()
                if not retval:
                    print("video capture fail")
                    break
                h, w, _ = frame.shape
                line_retval, L_lines, R_lines = line.line_detect(frame)

                # =========================선을 찾지 못했다면, 다음 프레임으로 continue=========================
                if line_retval == False:
                    self.sendBase64(frame)
                    continue

                # ===================================================================================
                # 고정 y 값
                y_fix = int(h * (2 / 3))

                # 화면 중앙 점
                center_x = int(w / 2)
                center_point = (center_x, y_fix)

                # 교점들을 저장할 리스트
                left_cross_points = []
                right_cross_points = []

                # 왼/오 선을 찾았는지 bool 변수에 저장
                L_lines_detected = bool(len(L_lines) != 0)
                R_lines_detected = bool(len(R_lines) != 0)

                # 둘다 찾았을 경우
                if L_lines_detected and R_lines_detected:
                    for each_line in L_lines:
                        x1, y1, x2, y2 = each_line
                        # 직선 그리기
                        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        # 직선의 기울기
                        slope = (y2 - y1) / (x2 - x1)
                        # 교점의 x 좌표
                        cross_x = ((y_fix - y1) / slope) + x1
                        
                        # 교점의 x 좌표 저장
                        left_cross_points.append(cross_x)

                    for each_line in R_lines:
                        x1, y1, x2, y2 = each_line
                        # 직선 그리기
                        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        # 직선의 기울기
                        slope = (y2 - y1) / (x2 - x1)
                        # 교점의 x 좌표
                        cross_x = ((y_fix - y1) / slope) + x1

                        # 교점의 x 좌표 저장
                        right_cross_points.append(cross_x)
                    
                    # 모든 선들의 가장 작은 x 좌표가 왼쪽, 큰 x 좌표가 오른쪽
                    left_line_x = min(left_cross_points)
                    right_line_x = max(right_cross_points)
                    
                    # 도로 너비의 반 계산 후 저장
                    road_half_width = (right_line_x - left_line_x) / 2
                    road_half_width_list.append(road_half_width)
                    
                    # 도로 중간 지점 저장
                    road_center_x = left_line_x + road_half_width
                    road_center_point = (int(road_center_x), y_fix)

                # 둘중 하나만 찾았을 경우
                elif L_lines_detected ^ R_lines_detected:
                    road_half_width = np.mean(road_half_width_list)

                    # 왼쪽 선만 찾았을 경우
                    if L_lines_detected:
                        for each_line in L_lines:
                            x1, y1, x2, y2 = each_line
                            # 직선 그리기
                            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            # 직선의 기울기
                            slope = (y2 - y1) / (x2 - x1)
                            # 교점의 x 좌표
                            cross_x = ((y_fix - y1) / slope) + x1

                            # 교점의 x 좌표 저장
                            left_cross_points.append(cross_x)
                        
                        # 왼쪽선들만 찾았으니, 그중 가장 작은 x 좌표가 왼쪽 선
                        left_line_x = min(left_cross_points)
                        # 도로 중간 지점 저장
                        road_center_x = left_line_x + road_half_width
                        road_center_point = (int(road_center_x), y_fix)

                    # 오른쪽 선만 찾았을 경우
                    else:
                        for each_line in R_lines:
                            x1, y1, x2, y2 = each_line
                            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            # 직선의 기울기
                            slope = (y2 - y1) / (x2 - x1)
                            # 교점의 x 좌표
                            cross_x = ((y_fix - y1) / slope) + x1

                            # 교점의 x 좌표 저장
                            right_cross_points.append(cross_x)

                        # 오른쪽선들만 찾았으니, 그중 가장 큰 x 좌표가 오른쪽 선
                        right_line_x = max(right_cross_points)
                        # 도로 중간 지점 저장
                        road_center_x = right_line_x - road_half_width
                        road_center_point = (int(road_center_x), y_fix)
                
                # 도로 중간 지점 / 자동차 중간 지점과 라인 시각화
                cv2.circle(frame, road_center_point, 5, (255, 0, 0), -1)
                cv2.circle(frame, center_point, 5, (0, 255, 0), -1)
                cv2.line(frame, road_center_point, center_point, (255, 255, 255), 2)

                # 왼쪽을 돌려야하면 음수, 오른쪽으로 돌려야하면 양수
                offset_width = road_center_x - center_x
                offset_height = h - y_fix

                # 각도 구하기
                # 오른쪽으로 회전해야 하는 경우 각도가 음수, 왼쪽으로 회전해야하는 경우 양수
                angle = np.arctan2(offset_height, offset_width) * 180 / (np.pi) - 90
                # print("before : {}".format(angle))

                angle = pid_controller.equation(angle)
                # print("after : {}".format(angle))

                # 핸들 제어
                self.ambulance.set_angle(angle)

                self.sendBase64(frame)
                # print("send")
            else:
                print("videoCapture is not opened")
                break
        self.client.loop_stop()

    def __on_connect(self, client, userdata, flags, rc):
        print("ImageMqttClient mqtt broker connect")
        # self.client.subscribe("command/camera/capture")

    def __on_disconnect(self, client, userdata, rc):
        print("ImageMqttClient mqtt broker disconnect")

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


if __name__ == '__main__':
    videoCapture = cv2.VideoCapture(0)
    videoCapture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    videoCapture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    imageMqttPublisher = ImageMqttPublisher("192.168.3.183", 1883, "/camerapub")
    imageMqttPublisher.connect()

    while True:
        if videoCapture.isOpened():
            retval, frame = videoCapture.read()
            if not retval:
                print("video capture fail")
                break
            imageMqttPublisher.sendBase64(frame)
        else:
            print("videoCapture is not opened")
            break

    imageMqttPublisher.disconnect()
    videoCapture.release()

