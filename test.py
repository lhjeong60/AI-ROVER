from mqtt.MqttClient import MqttClient
from mqtt.CameraMqttClient import ImageMqttPublisher
from ambulance.ambulance import Ambulance

car = Ambulance()

mqttClient = MqttClient("192.168.3.132", 1883, "command/#", "ambulance/battery/status", car)
mqttClient.start()
print("MqttClient start")

cameraClient =ImageMqttPublisher("192.168.3.132", 1883, "ambulance/camera/frameLine", car)
cameraClient.start()
print("CameraMqttClient start")