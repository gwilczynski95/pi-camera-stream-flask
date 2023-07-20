# Taken from: https://automaticaddison.com/motion-detection-using-opencv-on-raspberry-pi-4/

from pathlib import Path
import time

import cv2
import numpy as np


def load_video(path):
    cap = cv2.VideoCapture(str(path))

    out = []
    ret = True
    while ret:
        ret, frame = cap.read()
        if frame is not None:
            out.append(frame)

    cap.release()

    return np.array(out)


class FrameAnalyser:
    def __init__(self, history=150, varThreshold=25, detectShadows=True, weight=0.5, thresh_val=127):
        self.mog2 = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=varThreshold, detectShadows=detectShadows
        )
        self.mog2_kernel = np.ones((20, 20), np.uint8)
        self.weight = weight
        self.thresh_val = thresh_val

        self._simple_adaptive_avg = None

    def perform_mog2(self, frame):
        fg_mask = self.mog2.apply(frame)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.mog2_kernel)
        fg_mask = cv2.medianBlur(fg_mask, 5)
        _, fg_mask = cv2.threshold(fg_mask, self.thresh_val, 255, cv2.THRESH_BINARY)

        fg_mask = cv2.dilate(fg_mask, None, iterations=2)

        return fg_mask

    def perform_simple_adaptive(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self._simple_adaptive_avg is None:
            self._simple_adaptive_avg = gray.copy().astype("float")

        cv2.accumulateWeighted(gray, self._simple_adaptive_avg, self.weight)
        frame_delta = cv2.absdiff(gray, cv2.convertScaleAbs(self._simple_adaptive_avg))
        _, thresh = cv2.threshold(frame_delta, self.thresh_val, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)

        return thresh


def main():
    path_to_video = Path("/Users/gwilczynski/projects/cats/pi-camera-stream-flask/scripts/recording.h264")
    video = load_video(path_to_video)

    analyser = FrameAnalyser(thresh_val=10)

    for frame in video:
        parsed_frame = analyser.perform_simple_adaptive(frame)
        # parsed_frame = analyser.perform_mog2(frame)
        # Display the resulting frame
        cv2.imshow("Frame", parsed_frame)
        # cv2.imshow("Frame", frame)

        # Wait for keyPress for 1 millisecond
        key = cv2.waitKey(1) & 0xFF

        # exit this loop

    # Close down windows
    cv2.destroyAllWindows()

    stop = 1


if __name__ == '__main__':
    main()
