import cv2

# gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera
# Defaults to 1280x720 @ 60fps
# Flip the image by setting the flip_method (most common values: 0 and 2)
# display_width and display_height determine the size of the window on the screen


class Camera:
    def __init__(self, cap_w=1280, cap_h=720, dp_w=1280, dp_h=720, fps=60, flip_method=0):
        self.__cap_w = cap_w
        self.__cap_h = cap_h
        self.__dp_w = dp_w
        self.__dp_h = dp_h
        self.__fps = fps
        self.__flip_method = flip_method
        self.__videoCapture = cv2.VideoCapture(self.__gstreamer_pipline(), cv2.CAP_GSTREAMER)

    def __gstreamer_pipline(self):
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
                    self.__cap_w,
                    self.__cap_h,
                    self.__fps,
                    self.__flip_method,
                    self.__dp_w,
                    self.__dp_h,
                )
        )

    def show_camera(self):
        if self.__videoCapture.isOpened():
            window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE)
            # window가 종료되면 cv2.getWindowProperty는 -1을 리턴
            while cv2.getWindowProperty("CSI Camera", 0) >= 0:
                ret_val, img = self.__videoCapture.read()
                cv2.imshow("CSI Camera", img)
                # This also acts as
                keyCode = cv2.waitKey(30) & 0xFF
                # Stop the program on the ESC key
                if keyCode == 27:
                    break
            self.__videoCapture.release()
            cv2.destroyAllWindows()
        else:
            print("Unable to open camera")

    def read(self):
        retval, frame = self.__videoCapture.read()
        return retval, frame

    def release(self):
        self.__videoCapture.release()




if __name__ == "__main__":
    camera = Camera()
    camera.show_camera()