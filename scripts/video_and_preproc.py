from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2

from scripts.motion_detection import FrameAnalyser


def main():
    camera = PiCamera()
    camera.resolution = (640, 480)
    camera.framerate = 24

    analyser = FrameAnalyser(thresh_val=10)

    raw_capture = PiRGBArray(camera, size=(640, 480))

    time.sleep(0.1)

    for img in camera.capture_continuous(raw_capture, format="rgb", use_video_port=True):
        frame = img.array

        frame = analyser.perform_simple_adaptive(frame)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        raw_capture.truncate(0)
        if key == ord("q"):
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
