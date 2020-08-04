from mqtt.MqttClient import MqttClient
from mqtt.CameraMqttClient2 import ImageMqttPublisher
from ambulance.ambulance import Ambulance
from utils.camera import Camera
import detection.detect.obj_detect as obj_detect
import video_capture as vc
import threading

car = Ambulance()

mqttClient = MqttClient("192.168.3.132", 1883, "command/#", "ambulance/battery/status", car)
mqttClient.start()
print("MqttClient start")

# cameraClient =ImageMqttPublisher("192.168.3.132", 1883, "ambulance/camera/frameLine", car)
# cameraClient.start()
# print("CameraMqttClient start")


# 캡쳐 (cv2.imshow하기 때문에 camera mqtt와 동시에 실행 불가)
# capture_thread = vc.Capture_thread()
# capture_thread.camara_init()
# capture_thread.start()


# 차선 인식 & 제어 -> 객체 감지 영상을 mqtt로 전송
camera = Camera(cap_w=320, cap_h=240, dp_w=320, dp_h=240, fps=10, flip_method=0)
client = obj_detect.ResultImageMqttClient("192.168.3.132", 1883, "ambulance/camera/frameLine", car, camera)
client.start()