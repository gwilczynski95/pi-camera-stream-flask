# Taken from: https://automaticaddison.com/motion-detection-using-opencv-on-raspberry-pi-4/
import abc
from pathlib import Path
import time

import cv2
import numpy as np


class SignalDetector:
    def __init__(self, alpha: float = 0., beta: float = 0.99, gamma: float = 0.99, sigma_factor: float = 18.,
                 warmup_length: int = 100, cooldown_length: int = 100):
        self._alpha = None
        self._beta = None
        self._gamma = None
        self._sigma_factor = None
        self._avg_signal = None
        self._mean_avg_signal = None
        self._std_avg_signal = None
        self._warmup = None
        self._warmup_length = None
        self._cooldown = None
        self._cooldown_length = None
        self._detected = None
        self.reset(alpha, beta, gamma, sigma_factor, warmup_length, cooldown_length)

    def reset(self, alpha: float = 0.0, beta: float = 0.99, gamma: float = 0.99, sigma_factor: float = 18.,
              warmup_length: int = 100, cooldown_length: int = 100):
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._sigma_factor = sigma_factor
        self._avg_signal = None
        self._mean_avg_signal = None
        self._std_avg_signal = None
        self._warmup = warmup_length
        self._warmup_length = warmup_length
        self._cooldown = 0
        self._cooldown_length = cooldown_length
        self._detected = False

    @abc.abstractmethod
    def update(self, *args):
        raise NotImplementedError()

    def _update_avg_signal(self, signal):
        if self._avg_signal is None:
            self._avg_signal = signal
        self._avg_signal = self._alpha * self._avg_signal + (1. - self._alpha) * signal

    def _update_mean_avg_signal(self, signal_to_mean_avg):
        if self._mean_avg_signal is None:
            self._mean_avg_signal = signal_to_mean_avg
        self._mean_avg_signal = self._beta * self._mean_avg_signal + (1. - self._beta) * signal_to_mean_avg

    def _update_std_avg_signal(self, deviation_signal):
        avg_signal_deviation = np.abs(deviation_signal - self._mean_avg_signal)
        if self._std_avg_signal is None:
            self._std_avg_signal = avg_signal_deviation
        self._std_avg_signal = self._gamma * self._std_avg_signal + (1 - self._gamma) * avg_signal_deviation

    def _check_final_conditions(self, condition_signal):
        # if self._warmup > 0 or self._cooldown > 0:
        #     if self._warmup > 0:
        #         self._warmup -= 1
        #     if self._cooldown > 0:
        #         self._cooldown -= 1
        #     self._fire_detected = False
        # else:
        avg_signal_lower_bound = self._mean_avg_signal - self._sigma_factor * self._std_avg_signal
        avg_signal_upper_bound = self._mean_avg_signal + self._sigma_factor * self._std_avg_signal

        self._detected = np.logical_or(
            condition_signal < avg_signal_lower_bound, condition_signal > avg_signal_upper_bound
        ).astype(np.uint8) * 255
        # if self._fire_detected:
        #     self._cooldown = self._cooldown_length

    @property
    def fire_detected(self):
        return self._detected


class AdaptiveDetector(SignalDetector):
    def __init__(self, alpha: float = 0.0, beta: float = 0.99, gamma: float = 0.99, sigma_factor: float = 18.0,
                 warmup_length: int = 100, cooldown_length: int = 100):
        self._prev_avg_signal = None
        super(AdaptiveDetector, self).__init__(alpha, beta, gamma, sigma_factor, warmup_length, cooldown_length)

    def reset(self, alpha: float = 0.0, beta: float = 0.99, gamma: float = 0.99, sigma_factor: float = 18.0,
              warmup_length: int = 100, cooldown_length: int = 100):
        self._prev_avg_signal = None
        super(AdaptiveDetector, self).reset(alpha, beta, gamma, sigma_factor, warmup_length, cooldown_length)

    def update(self, frame: np.ndarray):
        self._update_avg_signal(frame)

        if self._prev_avg_signal is None:
            self._prev_avg_signal = self._avg_signal
        avg_signal_gradient = self._avg_signal - self._prev_avg_signal
        self._prev_avg_signal = self._avg_signal

        self._update_mean_avg_signal(avg_signal_gradient)
        self._update_std_avg_signal(avg_signal_gradient)
        self._check_final_conditions(avg_signal_gradient)
        return self._detected


class MovementDetector:
    def __init__(self, detector, fraction: float = 0.02, blur_ker_sz: tuple = (21, 21), dilation_iters: int = 2):
        self.detector = detector
        self.fraction = fraction
        self.blur_ker_sz = blur_ker_sz
        self.dilation_iters = dilation_iters

    def analyse(self, frame: np.ndarray):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.blur_ker_sz is not None:
            gray = cv2.GaussianBlur(gray, self.blur_ker_sz, 0).copy().astype("float")

        detected = self.detector.update(gray)
        if self.dilation_iters:
            detected = cv2.dilate(detected, None, iterations=self.dilation_iters)

        # todo: check if sum(bool) is ok
        return np.sum(detected.astype(bool)) >= self.fraction * np.prod(detected.shape)


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

    # analyser = FrameAnalyser(thresh_val=10)
    analyser = AdaptiveDetector()
    for frame in video:
        # parsed_frame = analyser.perform_simple_adaptive(frame)
        # parsed_frame = analyser.perform_mog2(frame)
        parsed_frame = analyser.update(frame)
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
