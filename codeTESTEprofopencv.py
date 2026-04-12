import cv2
import numpy as np
import requests

#url = "http://10.216.23.205/1024x768.mjpeg"
url = "http://10.216.23.143/1024x768.mjpeg"


r = requests.get(url, stream=True, timeout=10)
r.raise_for_status()

buf = b""
for chunk in r.iter_content(chunk_size=4096):
    buf += chunk
    a = buf.find(b"\xff\xd8")
    b = buf.find(b"\xff\xd9")
    if a != -1 and b != -1 and b > a:
        jpg = buf[a:b+2]
        buf = buf[b+2:]
        frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        cv2.imshow("ESP32-CAM", frame)
        if cv2.waitKey(1) == ord("q"):
            break

cv2.destroyAllWindows()