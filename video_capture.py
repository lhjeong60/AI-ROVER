import datetime
import cv2
import threading

class Capture_thread:
    def __init__(self):
        self.capture = None
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.record = False

    def gstreamer_pipline(self, cap_w, cap_h, fps, flip_method, dp_w, dp_h):
        return (
                "nvarguscamerasrc ! "
                "video/x-raw(memory:NVMM), "
                "width=(int)%d, height=(int)%d, "
                "format=(string)NV12, framerate=(fraction)%d/1 ! "
                "nvvidconv flip-method=%d ! "
                "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
                "videoconvert ! "
                "video/x-raw, format=(string)BGR ! appsink"
                % (
                    cap_w,
                    cap_h,
                    fps,
                    flip_method,
                    dp_w,
                    dp_h,
                )
        )


    def camara_init(self):
        self.capture = cv2.VideoCapture(self.gstreamer_pipline(320, 240, 60, 0, 320, 240), cv2.CAP_GSTREAMER)

    def run(self):
        while True:
            if (self.capture.get(cv2.CAP_PROP_POS_FRAMES) == self.capture.get(cv2.CAP_PROP_FRAME_COUNT)):
                self.capture.open(self.gstreamer_pipline(320, 240, 60, 0, 320, 240), cv2.CAP_GSTREAMER)

            ret, frame = self.capture.read()
            cv2.imshow("VideoFrame", frame)

            now = datetime.datetime.now().strftime("%d_%H-%M-%S")
            key = cv2.waitKey(33)

            if key == 27:
                break
            elif key == 67: # C (Capture)
                print("캡쳐")
                cv2.imwrite("/home/jetson/MyWorkspace/capture/img/" + str(now) + ".png", frame)
            elif key == 82: # R (Record)
                print("녹화 시작")
                self.record = True
                video = cv2.VideoWriter("/home/jetson/MyWorkspace/capture/video/" + str(now) + ".avi", self.fourcc, 20.0, (frame.shape[1], frame.shape[0]))
            elif key == 83: # S (Stop)
                print("녹화 중지")
                self.record = False
                video.release()

            if self.record == True:
                print("녹화 중..")
                video.write(frame)

        self.capture.release()
        cv2.destroyAllWindows()

    def start(self):
        thread = threading.Thread(target=self.run, daemon=True)
        thread.start()