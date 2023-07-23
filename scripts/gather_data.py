import argparse
import datetime
from pathlib import Path
import time

import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import pytz
import yaml

from scripts.motion_detection import AdaptiveDetector, MovementDetector


def _load_yaml(path):
    with open(path, "r") as stream:
        data = yaml.safe_load(stream)
    return data


def current_time(tz="Europe/Warsaw"):
    return datetime.now(pytz.timezone(tz))


def create_detector(detector_cfg: dict):
    if detector_cfg["name"] == "adaptive":
        minor_alg = AdaptiveDetector(**detector_cfg["minor_kwargs"])
    else:
        raise NotImplementedError(f"Algorithm {detector_cfg['name']} not implemented")
    main_detector = MovementDetector(
        **{
            "detector": minor_alg,
            **detector_cfg["major_kwargs"]
        }
    )
    return main_detector


def init_camera(config: dict):
    camera = PiCamera(
        resolution=config["resolution"],
        framerate=config["frame_rate"]
    )
    raw_capture = PiRGBArray(
        camera,
        size=config["resulotion"]
    )
    return camera, raw_capture


def create_out_filename(extension, seconds_before, strf="%Y-%m-%d_%H:%M:%S"):
    curr_datetime = current_time() - datetime.timedelta(seconds=seconds_before)
    datetime_str = str(curr_datetime.strftime(strf))
    return f"{datetime_str}{extension}"


def main(config: dict):
    output_path = config["output_path"]
    output_path.mkdir(parents=True, exist_ok=True)

    # init detector and camera
    detector = create_detector(config["detection_algorithm"])

    frames_before = int(config["seconds_before"] / 60. * config["frame_rate"])
    frames_after = int(config["seconds_after"] / 60. * config["frame_rate"])

    before_buffer = np.zeros(
        (frames_before,) + config["resolution"] + (3,), dtype=np.uint8
    )

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')

    camera, raw_capture = init_camera(config)

    time.sleep(0.1)

    start_clip = False
    in_bf_buff_state = 0  # if value == len(before_buffer) then it's full
    in_after_buffer_state = frames_after  # if value == 0 then save what was recorded

    for img in camera.capture_continuous(raw_capture, format="rgb", use_video_port=True):
        frame = img.array
        if in_bf_buff_state < before_buffer.shape[0]:
            before_buffer[
                in_bf_buff_state, :, :, :
            ] = frame
            in_bf_buff_state += 1
        else:
            is_detected = detector.analyse(frame)
            if is_detected:
                if not start_clip:
                    start_clip = True
                    filename = create_out_filename(
                        config["video_extension"],
                        config["seconds_before"]
                    )
                    clip_obj = cv2.VideoWriter(
                        filename,
                        fourcc,
                        config["frame_rate"],
                        config["resolution"]
                    )
                    for _frame in before_buffer:
                        clip_obj.write(frame)
                in_after_buffer_state = frames_after
                clip_obj.write(frame)
            elif not is_detected and in_after_buffer_state >= 0 and start_clip:
                in_after_buffer_state -= 1
                clip_obj.append(frame)
                if in_after_buffer_state == -1:
                    start_clip = False
                    clip_obj.release()
            else:
                before_buffer = np.roll(before_buffer, 1, axis=0)
                before_buffer[
                    in_bf_buff_state, :, :, :
                ] = frame


if __name__ == '__main__':
    cfg = _load_yaml("scripts/gather_data.yaml")
    main(cfg)
