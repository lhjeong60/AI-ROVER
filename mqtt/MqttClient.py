import paho.mqtt.client as mqtt
import threading
import time
import json
from ambulance.ambulance import Ambulance

class MqttClient:
    def __init__(self, brokerip=None, brokerport=1883, subtopic=None, ambulance=None):
        self.__brokerip = brokerip
        self.__brokerport = brokerport
        self.__subtopic = subtopic
        self.__client = mqtt.Client()
        self.__client.on_connect = self.__on_connect
        self.__client.on_disconnect = self.__on_disconnect
        self.__client.on_message = self.__on_message
        self.__ambulance = ambulance
        self.__stop = False

    def __on_connect(self, client, userdata, flags, rc):
        print("** connection **")
        self.__client.subscribe(self.__subtopic)
        self.__client.subscribe("car/1/destination")


    def __on_disconnect(self, client, userdata, rc):
        print("** disconnection **")

    def __on_message(self, client, userdata, message):
        if "backTire" in message.topic:
            if "forward" in message.topic:
                self.__ambulance.set_speed(0.6)
                self.__ambulance.forward()
            elif "backward" in message.topic:
                self.__ambulance.set_speed(0.6)
                self.__ambulance.backward()
            elif "stop" in message.topic:
                self.__ambulance.stop()

        elif "frontTire" in message.topic:
            if "left" in message.topic:
                self.__ambulance.handle_left()
            elif "right" in message.topic:
                self.__ambulance.handle_right()
            elif "front" in message.topic:
                self.__ambulance.handle_refront()

        elif "changemode" in message.topic:
            if "auto" in message.topic:
                self.__ambulance.set_mode(Ambulance.AUTO_MODE)
                self.__ambulance.set_max_speed(0.55)
            if "manual" in message.topic:
                self.__ambulance.set_mode(Ambulance.MANUAL_MODE)

        elif "road" in message.topic:
            if "change" in message.topic:
                self.__ambulance.change_road()

        elif "process" in message.topic:
            # print(message.topic)
            if "stop" in message.topic:
                self.__ambulance.oled_thread_stop()
                self.__stop = True
                self.__client.disconnect()

        elif message.topic == "car/1/destination":
            dst = str(message.payload, encoding="UTF-8")
            self.__ambulance.set_dst(dst)
            self.__ambulance.set_working(True)
            self.__ambulance.set_mode(Ambulance.AUTO_MODE)



    def __run(self):
        self.__client.connect(self.__brokerip, self.__brokerport)
        self.__client.loop_start()
        while not self.__stop:
            self.__client.publish("ambulance/1/status", json.dumps(self.__ambulance.get_status()))
            time.sleep(0.2)
        self.__client.loop_stop()



    def start(self):
        thread = threading.Thread(target=self.__run)
        thread.start()

    def stop(self):
        self.__client.unsubscribe(self.__subtopic)
        self.__client.disconnect()


