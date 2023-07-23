import time
import cv2
import numpy as np

from scripts.motion_detection import FrameAnalyser, AdaptiveDetector


def main():
    scale_percent = 0.7
    cap = cv2.VideoCapture(0)

    analyser = AdaptiveDetector()

    time.sleep(0.1)

    while cap.isOpened():
        ret, frame = cap.read()
        width = int(frame.shape[1] * scale_percent)
        height = int(frame.shape[0] * scale_percent)
        dim = (width, height)
        frame = cv2.resize(frame, dim)

        frame = analyser.update(frame)
        mean_avg_signal = analyser._mean_avg_signal
        std_avg_signal = analyser._std_avg_signal

        full_frame = np.hstack(
            [
                mean_avg_signal.astype(np.uint8),
                std_avg_signal.astype(np.uint8),
                frame
            ]
        )

        cv2.imshow("Frame", full_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
