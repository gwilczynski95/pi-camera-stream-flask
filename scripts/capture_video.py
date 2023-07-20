from picamera import PiCamera
from time import sleep


def main():
    sleep_time_s = 20
    out_path = "recording.h264"

    camera = PiCamera()

    camera.start_preview()
    camera.start_recording(out_path, format="h264")
    sleep(sleep_time_s)
    camera.stop_recording()
    camera.stop_preview()


if __name__ == '__main__':
    main()
