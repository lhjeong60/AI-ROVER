import time
import os
import sys
import threading
import numpy as np
from collections import deque
from datetime import datetime

from jetracer.nvidia_racecar import NvidiaRacecar

current_path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(current_path)

from utils.camera import Camera
import utils.ipaddress as ip
import utils.power as pw
from utils.display import Oled
import line_detect.line_detect as line_detect
from utils.pid_controller import PIDController
from sensor.distance import Distance
from sensor.pcf8591 import Pcf8591

class Ambulance:
    AUTO_MODE = 1
    MANUAL_MODE = 2

    def __init__(self):
        # 센서 객체 생성
        self.__pcf8591 = Pcf8591(0x48)
        self.__distance = Distance(self.__pcf8591)
        self.pre = 0

        # 모터 제어
        self.__motor_control = NvidiaRacecar()
        self.__stop_flag = False


        # Oled 표시, 스레딩
        self.__oled = Oled()
        self.__oled_flag = True
        self.__oled_thread = threading.Thread(target=self.__oled_setting, daemon=True)
        self.__oled_thread.start()

        # =================motor values========================
        self.__handle_angle = 0
        self.__dcMotor_speed = 0
        self.__motor_direction = "stop"

        # =================car status==========================
        self.__working = False

        # =================pid controller======================
        self.pid_controller = PIDController(round(datetime.utcnow().timestamp() * 1000))
        # self.pid_controller.set_gain(0.63, -0.001, 0.23)
#        self.pid_controller.set_gain(0.61, 0, 0.22)
        self.pid_controller.set_gain(0.55, 0, 0.23)
        # ==================line detect variable===============
        self.road_half_width_list = deque(maxlen=10)
        self.road_half_width_list.append(100)

        self.prev_road_center_x = deque(maxlen=5)
        self.prev_road_center_x.append(120)

        self.which_side = None
        self.change_road_flag = False
        self.init_flag = None

        self.L_lines = []
        self.R_lines = []

        self.L_x = 0
        self.R_x = 240

        # =================mode 1:auto 2:manual================
        self.__mode = 0

        # =======================count============================
        self.count = 0
    def __oled_setting(self):
        while self.__oled_flag:
            self.__oled.set_text(ip.get_ip_address_wlan0() + "\n" + pw.get_power_status())
            time.sleep(2)

    def get_voltage_percentage(self):
        voltage_percentage = round((pw.get_voltage() - 11.1) / 0.015)
        return voltage_percentage

    # ============================핸들 원격 제어==============================
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

    # =============================속도 원격 제어=============================

    def backward(self, speed):
        self.__motor_control.throttle_gain = speed
        self.__motor_control.throttle = 1
        self.__dcMotor_speed = speed
        self.__motor_direction = "backward"

    def forward(self, speed):
        if self.__stop_flag:
            self.__motor_control.throttle = 0
            self.__motor_control.throttle_gain = 0.9
            self.__motor_control.throttle = -1
            self.__stop_flag = False
        self.__motor_control.throttle = 0
        self.__motor_control.throttle_gain = speed
        self.__motor_control.throttle = -1
        self.__dcMotor_speed = speed
        self.__motor_direction = "forward"

    def stop(self):
        # print("stop")
        self.__dcMotor_speed = 0
        self.__motor_control.throttle_gain = self.__dcMotor_speed
        self.__motor_control.throttle = self.__dcMotor_speed
        self.__stop_flag = True
        self.__motor_direction = "stop"
    # ===========================핸들 각도 제어(자율 주행)==========================

    def set_angle(self, angle):
        if angle > 30:
            angle = 30
        if angle < -30:
            angle = -30
        steering = angle / 30

        self.__motor_control.steering = steering
        self.__handle_angle = steering

    # ===========================자율 주행 =====================================
    def set_mode(self, mode):
        self.__mode = mode

    def auto_drive(self, frame, flag, crosswalk_flag):
        if self.init_flag is None:
            temp_birdeye, M, Minv = line_detect.birdeye(line_detect.img_preprocessing(frame))
            crosswalk_flag, self.which_side = line_detect.get_crosswalk_flag(temp_birdeye), line_detect.get_which_side(temp_birdeye)
            self.init_flag = -1
        
        # 차선 검출
        if flag == 0:
            crosswalk_flag, line_retval, self.L_lines, self.R_lines = line_detect.line_detect(frame)

            # 차선 검출이 안 되었을 때
            if line_retval == False:
                return -1, False
            else:
                return 1, crosswalk_flag
        
        # 차선 검출 후 offset 계산 및 제어
        if flag == 1:
            # 차선 변경을 하지 않을 때
            if not self.change_road_flag:
                angle, self.L_x, self.R_x = line_detect.offset_detect(frame, crosswalk_flag, self.which_side, self.L_lines, self.R_lines, self.L_x, self.R_x, self.road_half_width_list, self.prev_road_center_x)
                
                # 자율 주행 모드일 때만 동작
                if self.__mode == self.AUTO_MODE:
                    angle = self.pid_controller.equation(angle)
                    self.set_angle(angle)
                    angle = abs(angle)
                    
                    # 회전할 각도의 절대량에 따라 속도를 감속
                    if angle < 10:
                        angle = 0.51 - (angle/200)
                    elif 10 <= angle < 20:
                        angle = 0.49
                    else:
                        angle = 0.47
                    
                    # 적외선 센서 사용
                    distance_stop_flag = False
                    cm = self.__distance.read()

                    # 많이 튀는 값 날리기
                    if abs(cm - self.pre) < 5:
                        # 최대 최소로 제한
                        if cm > 80:
                            cm = 80
                        elif cm < 0:
                            cm = 0

                        # 완전히 가까워지면 확실하게 정지
                        if cm <= 25:
                            self.stop()
                            self.backward(0.4)
                            distance_stop_flag = True
                        # 가까운 물체가 감지되면 정지 시작
                        elif 25 < cm <= 35:
                            self.stop()
                            self.backward(0.25)

                        # 물체가 감지되지 않으면 그냥 주행
                        else:
                            if distance_stop_flag:
                                self.forward(1.0)
                                distance_stop_flag = False
                            self.forward(angle)
                    self.pre = cm

                return 2, crosswalk_flag

            # 차선 변경 시 동작
            else:
                # 오른쪽 차선에 있을 때
                if not self.which_side:
                    if self.count < 16:
                        # 핸들 왼쪽으로 꺾
                        self.set_angle(16)
                        self.forward(0.56)
                        self.count += 1

                    else:
                        print("yeah")
                        if bool(len(self.L_lines) != 0) or bool(len(self.R_lines) != 0):
                            self.change_road_flag = False
                            self.count = 0

                # 왼쪽 차선에 있을 때
                else:
                    if self.count < 16:
                        # 핸들 오른쪽으로 꺾
                        self.set_angle(-16)
                        self.forward(0.56)
                        self.count += 1

                    else:
                        print("yeah")
                        if bool(len(self.L_lines)) != 0 or bool(len(self.R_lines)) != 0:
                            self.change_road_flag = False
                            self.count = 0

                return 2, False

    # ===========================차선 상태 변경=================================
    def change_road(self):
        self.change_road_flag = True
        if self.which_side is not None:
            self.which_side = not self.which_side
        self.print_which_side()
    
    # 차선 위치 출력
    def print_which_side(self):
        if self.which_side is None:
            print("which_side : None")

        elif self.which_side:
            print("which_side : R")
        else:
            print("which_side : L")


    # ===========================스레드 중단====================================
    def oled_thread_stop(self):
        self.__oled_flag = False
        self.__oled_thread.join()

    # ===========================운송 중 설정 ==================================
    def set_working(self, work):
        self.__working = work

    def get_status(self):
        status = {}
        status["battery"] = self.get_voltage_percentage()
        status["angle"] = self.__handle_angle
        status["speed"] = self.__dcMotor_speed
        status["direction"] = self.__motor_direction
        status["mode"] = self.__mode
        status["working"] = self.__working

        return status