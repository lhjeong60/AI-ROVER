import time
import os
import sys
import threading

from jetracer.nvidia_racecar import NvidiaRacecar

current_path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(current_path)

from utils.camera import Camera
import utils.ipaddress as ip
import utils.power as pw
from utils.display import Oled


class Ambulance:
    def __init__(self):
        self.__motor_control = NvidiaRacecar()

        # Oled 표시, 스레딩
        self.__oled = Oled()
        self.__oled_thread = threading.Thread(target=self.__oled_setting, daemon=True)
        self.__oled_thread.start()
        # =================motor values========================
        self.__handle_angle = 0
        self.__dcMotor_speed = 0

    def __oled_setting(self):
        while True:
            self.__oled.set_text(ip.get_ip_address_wlan0() + "\n" + pw.get_power_status())
            time.sleep(2)

    def get_voltage_percentage(self):
        voltage_percentage = round((pw.get_voltage() - 11.1) / 0.015)
        return voltage_percentage

    # def cam_read(self):
    #     retval, frame = self.__camera.read()
    #     return retval, frame

    # ==========================================================================
    def handle_right(self):
        if self.__handle_angle >= -1:
            self.__handle_angle -= 0.1
            self.__motor_control.steering = self.__handle_angle
            # print("right:" + self.__handle_angle)

    def handle_left(self):
        if self.__handle_angle <= 1:
            self.__handle_angle += 0.1
            self.__motor_control.steering = self.__handle_angle
            # print("left : " + self.__handle_angle)

    def handle_refront(self):
        while True:
            if -0.05 <= self.__handle_angle <= 0.05:
                self.__handle_angle = 0
                self.__motor_control.steering = self.__handle_angle
                break


            if self.__handle_angle < 0:
                self.__handle_angle += 0.05
            else:
                self.__handle_angle -= 0.05
            # print(self.__handle_angle)
            self.__motor_control.steering = self.__handle_angle

    def backward(self):
        # if self.__dcMotor_speed < 1.0:
        #     self.__dcMotor_speed += 0.01
        # self.__motor_control.throttle_gain = self.__dcMotor_speed
        # print("forward")
        self.__motor_control.throttle = 0
        self.__motor_control.throttle = 1
        self.__motor_control.throttle_gain = 0.7

    def forward(self):
        # if self.__dcMotor_speed < 1.0:
        #     self.__dcMotor_speed += 0.01
        # self.__motor_control.throttle_gain = self.__dcMotor_speed
        # print("backward")
        self.__motor_control.throttle = 0
        self.__motor_control.throttle = -1
        self.__motor_control.throttle_gain = 0.7

    def stop(self):
        # print("stop")
        self.__dcMotor_speed = 0
        self.__motor_control.throttle_gain = self.__dcMotor_speed
        self.__motor_control.throttle = self.__dcMotor_speed
    # =========================================================================

    def set_angle(self, angle):
        if angle > 30:
            angle = 30
        if angle < -30:
            angle = -30
        steering = angle / 30
        self.__motor_control.steering = steering

