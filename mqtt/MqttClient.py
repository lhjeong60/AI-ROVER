import paho.mqtt.client as mqtt
import threading
import time

class MqttClient:
    def __init__(self, brokerip=None, brokerport=1883, subtopic=None, pubtopic=None ,ambulance=None):
        self.__brokerip = brokerip
        self.__brokerport = brokerport
        self.__subtopic = subtopic
        self.__pubtopic = pubtopic
        self.__client = mqtt.Client()
        self.__client.on_connect = self.__on_connect
        self.__client.on_disconnect = self.__on_disconnect
        self.__client.on_message = self.__on_message
        self.__ambulance = ambulance
        self.__stop = False

    def __on_connect(self, client, userdata, flags, rc):
        print("** connection **")
        self.__client.subscribe(self.__subtopic)


    def __on_disconnect(self, client, userdata, rc):
        print("** disconnection **")

    def __on_message(self, client, userdata, message):
        if "backTire" in message.topic:
            if "forward" in message.topic:
                self.__ambulance.forward()
            elif "backward" in message.topic:
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


    def __run(self):
        self.__client.connect(self.__brokerip, self.__brokerport)
        self.__client.loop_start()
        while not self.__stop:
            self.__client.publish(self.__pubtopic ,self.__ambulance.get_voltage_percentage())
            time.sleep(1)
        self.__client.loop_stop()



    def start(self):
        thread = threading.Thread(target=self.__run)
        thread.start()

    def stop(self):
        self.__client.unsubscribe(self.__subtopic)
        self.__client.disconnect()


