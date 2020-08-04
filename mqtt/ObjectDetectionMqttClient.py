import time
import cv2
import paho.mqtt.client as mqtt
import threading
import base64

import os
import sys

current_path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(current_path)

from utils.camera import Camera

class ObjectDetectionMqttClient:
    def __init__(self, brokerIp=None, brokerPort=1883, pubTopic=None, ambulance=None):
        self.brokerIp = brokerIp
        self.brokerPort = brokerPort
        self.pubTopic = pubTopic
        self.client = mqtt.Client()
        self.client.on_connect = self.__on_connect
        self.client.on_disconnect = self.__on_disconnect

        self.camera = Camera(cap_w=320, cap_h=240, dp_w=320, dp_h=240, fps=10, flip_method=0)
        self.camera.camera_init()

    def start(self):
        thread = threading.Thread(target=self.__run, daemon=True)
        thread.start()

    def __run(self):
        self.client.connect(self.brokerIp, self.brokerPort)



    def __on_connect(self, client, userdata, flags, rc):
        print("ImageMqttClient mqtt broker connect")

    def __on_disconnect(self, client, userdata, rc):
        print("ImageMqttClient mqtt broker disconnect")

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