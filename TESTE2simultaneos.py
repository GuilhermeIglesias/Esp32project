import cv2
import numpy as np
import requests
import threading

#url = "http://10.216.23.205/1024x768.mjpeg"
#url = "http://10.216.23.143/1024x768.mjpeg"

URL_CAM1 = "http://10.175.42.205/1024x768.mjpeg"
URL_CAM2 = "http://10.175.42.43/1024x768.mjpeg"  # IP da segunda ESP32

frame_cam1 = None
frame_cam2 = None
lock1 = threading.Lock()
lock2 = threading.Lock()

def read_stream(url, frame_ref, lock):
    global frame_cam1, frame_cam2
    r = requests.get(url, stream=True, timeout=10)
    r.raise_for_status()
    buf = b""
    for chunk in r.iter_content(chunk_size=4096):
        buf += chunk
        a = buf.find(b"\xff\xd8")
        b_end = buf.find(b"\xff\xd9")
        if a != -1 and b_end != -1 and b_end > a:
            jpg = buf[a:b_end+2]
            buf = buf[b_end+2:]
            frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                with lock:
                    if frame_ref == 1:
                        frame_cam1 = frame
                    else:
                        frame_cam2 = frame

t1 = threading.Thread(target=read_stream, args=(URL_CAM1, 1, lock1), daemon=True)
t2 = threading.Thread(target=read_stream, args=(URL_CAM2, 2, lock2), daemon=True)
t1.start()
t2.start()

while True:
    with lock1:
        f1 = frame_cam1.copy() if frame_cam1 is not None else None
    with lock2:
        f2 = frame_cam2.copy() if frame_cam2 is not None else None

    if f1 is not None and f2 is not None:
        combined = np.hstack((f1, f2))  # mostra lado a lado
        cv2.imshow("Stereo ESP32-CAM", combined)

    if cv2.waitKey(1) == ord("q"):
        break

cv2.destroyAllWindows()