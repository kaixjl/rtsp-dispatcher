import sys
from collections import namedtuple
import threading
import subprocess
import multiprocessing as mp
import queue
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
            *params,
            '-f', 'rtsp', #  flv rtsp
            '-rtsp_transport', 'tcp', 
            self.url] # rtsp rtmp

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
        if not self.opened:
            self.open()
        self.writer.stdin.write(img.tobytes())
        return True

class RtspDispatcher(threading.Thread):
    def __init__(self, config="rtsp-dispatcher.yaml", capacity=50, q=None):
        # type: (str | dict, int, None | mp.Queue) -> None
        super(RtspDispatcher, self).__init__()
        if isinstance(config, str):
            with open(config, "r") as f:
                config = yaml.safe_load(f)
        self.config = config
        self.capacity = capacity

        if q is not None:
            self.queue = q # type: mp.Queue[dict[str]]
        else:
            self.queue = mp.Queue(maxsize=self.capacity) # type: mp.Queue[dict[str]]

        self.cams = { it["name"]: RtspPusherFF(**{ k:v for k,v in it.items() if k!="name"}) for it in self.config["servers"] }

        # === multi-thread ===
        self.lock = threading.Lock()
        self.running = False
        # --- multi-thread ---

    def push(self, cam, img, block=True, timeout=None):
        # type: (str, np.ndarray, bool, float | None) -> bool
        """
        # Args
        - cam: cam name
        - img: opencv mat
        """
        if cam not in self.cams:
            print(f"Camera {cam} not found.", file=sys.stderr)
            return False
        try:
            self.queue.put(dict(cam=cam, img=img), block=block, timeout=timeout)
            return True
        except Exception as e:
            print(f"Received Exception while enqueing ({e})", file=sys.stderr)
            return False

    def process_one(self, block=True, timeout=None):
        # type: (bool, float | None) -> bool
        try:
            msg = self.queue.get(block, timeout)
            return self.cams[msg["cam"]].write(msg["img"])
        except Exception as e:
            print(f"Received Exception while processing ({e})", file=sys.stderr)
            return False

    # === multi-thread ===

    def run(self):
        self.running = True
        currtime = time.time()
        while self.running:
            if not self.running: break
            self.process_one()
            print(f"pull time per frame: {(time.time()-currtime)*1000:.2f}ms")
            currtime = time.time()

    def stop(self):
        self.running = False

    # --- multi-thread ---


def dispatch_rtsp(q):
    dispatcher = RtspDispatcher("rtsp-dispatcher.yaml", q=q)
    dispatcher.run()


def main():
    q = mp.Queue(50)
    subproc = mp.Process(target=dispatch_rtsp, args=(q,))
    subproc.start()

    fps = 25
    i = 0
    cams = ["cam0", "cam1", "cam2", "cam3"]
    starts = [0, 45, 90, 135]
    lengths = [45, 45, 45, 45]
    while True:
        currtime = time.time()
        i = (i + 1) % (fps * 5) # 0 ~ (fps * 5 - 1)
        for j in range(len(cams)):
            img = np.ones((1080, 1920, 3), dtype=np.uint8) * 255
            img[:,:,0] = starts[j] + int(i / (fps * 5 - 1) * lengths[j])
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
            q.put(dict(cam=cams[j], img=img))
        difftime = time.time() - currtime
        print(f"push difftime: {difftime*1000:.2f}ms")
        time.sleep(max(1/fps - difftime, 0))
        print(f"push time per frame: {(time.time()-currtime)*1000:.2f}ms")

    dispatcher.stop()
    dispatcher.join()
    pass

if __name__ == "__main__":
    main()
