import cv2
import paho.mqtt.client as mqtt
import threading
import base64

import numpy as np
import os
import sys

current_path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(current_path)

from utils.camera import Camera
import line_detect.main_test as line

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
        self.client.loop_start()
        while True:
            if self.camera.isOpened():
                retval, frame = self.camera.read()
                if not retval:
                    print("video capture fail")
                    break
                h, w, _ = frame.shape
                lines = line.line_detect(frame)

                # =========================선을 찾지 못했다면, 다음 프레임으로 continue=========================
                if lines is None:
                    self.sendBase64(frame)
                    continue

                # ===================================================================================
                # 고정 y 값
                y_fix = int(h * (2 / 3))
                # 교점들을 저장할 리스트
                cross_points = []
                
                for each_line in lines:
                    x1, y1, x2, y2 = each_line
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
                    
                    # 직선의 기울기
                    slope = (y2 - y1) / (x2 - x1)
                    # 교점의 x 좌표
                    cross_x = ((y_fix - y1) / slope) + x1

                    cross_points.append(cross_x)

                left_line_x = min(cross_points)
                right_line_x = max(cross_points)
                center_x = int(w/2)
                road_center_x = left_line_x + (right_line_x - left_line_x) / 2
                road_center_point = (int(road_center_x), y_fix)
                center_point = (center_x, y_fix)
                
                # 포인트와 라인 시각화
                cv2.circle(frame, road_center_point, 5, (255, 0, 0), -1)
                cv2.circle(frame, center_point, 5, (0, 255, 0), -1)
                cv2.line(frame, road_center_point, center_point, (255, 255, 255), 2)

                # 왼쪽을 돌려야하면 음수, 오른쪽으로 돌려야하면 양수
                offset_width = road_center_x - center_x
                offset_height = h - y_fix

                # 각도 구하기
                # 오른쪽으로 회전해야 하는 경우 각도가 음수, 왼쪽으로 회전해야하는 경우 양수
                angle = np.arctan2(offset_height, offset_width) * 180 / (np.pi) - 90
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

    # def __on_message(self, client, userdata, message):
    #     if "capture" in message.topic:
    #         retval, frame = self.camera.videoCapture.read()
    #         if retval:
    #             img = np.copy(frame)
    #             cv2.imwrite("/home/pi/Project/SensingRover/capture/capture_image" + str(self.__cnt) + ".jpg", img)
    #             self.__cnt += 1
    #             capval, bytes = cv2.imencode(".jpg", frame)
    #             if capval:
    #                 cap_b64_bytes = base64.b64encode(bytes)
    #                 self.client.publish("/capturepub", cap_b64_bytes)
    #                 print("pub complete")


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

