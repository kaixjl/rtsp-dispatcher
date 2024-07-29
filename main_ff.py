import sys
from collections import namedtuple, deque
import threading
import subprocess
import time
import yaml
import cv2
import numpy as np


class RtspPusherFF:
    def __init__(self, url, width, height, fps, encoder, **kwargs):
        """
        # Args
        - encoder: see `ffmpeg -encoders`
        - kwargs: params of encoder, see `ffmpeg -h encoder=<YOUR ENCODER>`, e.g. bitrate, profile, preset, tune.
        """
        self.url = url
        self.width = width
        self.height = height
        self.fps = fps
        self.encoder = encoder
        self.writer = None

        params = []
        for k, v in kwargs.items():
            params.append(f"-{k}")
            params.append(v)

        self.command = ['ffmpeg',
            '-y', '-an',
            '-f', 'rawvideo',
            '-vcodec','rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', "{}x{}".format(self.width, self.height),
            '-r', str(self.fps),
            '-i', '-',
            '-c:v', self.encoder,
            '-pix_fmt', 'yuv420p',
            '-preset', 'ultrafast',
            *params,
            '-f', 'rtsp', #  flv rtsp
            '-rtsp_transport', 'tcp', 
            self.url] # rtsp rtmp
        self.open()

    def __del__(self):
        self.close()

    def open(self):
        self.close()
        self.writer = subprocess.Popen(self.command, shell=False, stdin=subprocess.PIPE)
    
    def close(self):
        if self.writer is not None:
            self.writer.terminate()
    
    @property
    def opened(self):
        return self.writer is not None and self.writer.poll() == None
    
    def write(self, img):
        # type: (cv2.Mat) -> bool
        if not self.opened: return False
        self.writer.stdin.write(img.tobytes())
        return True

class RtspDispatcher(threading.Thread):
    MsgType = namedtuple("MsgType", ["cam", "img"])
    def __init__(self, config="rtsp-dispatcher.yaml", capacity=50):
        super(RtspDispatcher, self).__init__()
        if isinstance(config, str):
            with open(config, "r") as f:
                config = yaml.safe_load(f)
        self.config = config
        self.capacity = capacity
        self.deque = deque(maxlen=self.capacity) # type: deque[RtspDispatcher.MsgType]
        self.cams = { it["name"]: RtspPusherFF(**{ k:v for k,v in it.items() if k!="name"}) for it in self.config["servers"] }

        # === multi-thread ===
        self.lock = threading.Lock()
        self.running = False
        # --- multi-thread ---

    def push(self, cam, img):
        # type: (str, np.ndarray) -> bool
        """
        # Args
        - cam: cam name
        - img: opencv mat
        """
        if cam not in self.cams:
            print(f"Camera {cam} not found.", file=sys.stderr)
            return False
        self.deque.append(RtspDispatcher.MsgType(cam=cam, img=img))
        return True

    def process_one(self):
        # type: () -> bool
        if len(self.deque) == 0: return False
        msg = self.deque.popleft()
        self.cams[msg.cam].write(msg.img)

    # === multi-thread ===
    def push_safe(self, slot, msg):
        self.lock.acquire(True, -1)
        self.push(slot, msg)
        self.lock.release()

    def process_one_safe(self):
        # type: () -> bool
        if len(self.deque) == 0: return False
        if self.lock.acquire(True, 1):
            self.process_one()
            self.lock.release()
            return True
        return False

    def run(self):
        self.running = True
        while self.running:
            if len(self.deque) == 0:
                time.sleep(0)
                continue
            if self.lock.acquire(False, -1):
                if not self.running: break
                self.process_one()
                self.lock.release()

    def stop(self):
        self.running = False
    # --- multi-thread ---


def main():
    dispatcher = RtspDispatcher("rtsp-dispatcher.yaml")
    dispatcher.start()
    fps = 25
    i = 0
    while True:
        i = (i + 1) % (fps * 5)
        img = np.ones((480, 640, 3), dtype=np.uint8) * 255
        img[:,:,0] = int(i / (fps * 5 - 1) * 180)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        dispatcher.push("cam0", img)
        time.sleep(1/fps)

    dispatcher.stop()
    dispatcher.join()
    pass

if __name__ == "__main__":
    main()
