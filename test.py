from mqtt.MqttClient import MqttClient
from mqtt.CameraMqttClient import ImageMqttPublisher
from ambulance.ambulance import Ambulance

car = Ambulance()

mqttClient = MqttClient("192.168.3.183", 1883, "command/#", car)
mqttClient.start()
print("MqttClient start")

cameraClient =ImageMqttPublisher("192.168.3.183", 1883, "/camerapub", car)
cameraClient.start()
print("CameraMqttClient start")