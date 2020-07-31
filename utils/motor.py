from jetracer.nvidia_racecar import NvidiaRacecar
import time

#%%
class Motor:
    def __init__(self):
        self.__motor = NvidiaRacecar()
    
    # 핸들 조절(각도)
    def steering(self, rotate_gain):
        # rotate_gain : [-1, 1] => 우(-1), 정면(0) ,좌(1)
        self.__motor.steering = rotate_gain
    
    # 속도 조절
    def throttle_gain(self, dc_gain):
        self.__motor.throttle_gain = dc_gain

    # 방향 지정
    def throttle(self, direction):
        # forward : -1, backward : 1
        self.__motor.throttle = direction


if __name__ == '__main__':
    motor = Motor()
    rotate_gain = -1
    for i in range(3):
        motor.steering(rotate_gain)
        time.sleep(1)
        rotate_gain += 1
    time.sleep(1)
    motor.steering(0)

    motor.throttle(1)
    motor.throttle_gain(0.5)
    time.sleep(2)

    motor.throttle(0)
    time.sleep(2)

    motor.throttle(-1)
    time.sleep(2)

    motor.throttle_gain(0)
    motor.throttle(0)
