from mqtt.MqttClient import MqttClient
from mqtt.CameraMqttClient import ImageMqttPublisher
from ambulance.ambulance import Ambulance
import video_capture as vc
import threading

car = Ambulance()

mqttClient = MqttClient("192.168.3.132", 1883, "command/#", "ambulance/battery/status", car)
mqttClient.start()
print("MqttClient start")

cameraClient =ImageMqttPublisher("192.168.3.132", 1883, "ambulance/camera/frameLine", car)
cameraClient.start()
print("CameraMqttClient start")


# 캡쳐 (cv2.imshow하기 때문에 camera mqtt와 동시에 실행 불가)
# capture_thread = vc.Capture_thread()
# capture_thread.camara_init()
# capture_thread.start()
