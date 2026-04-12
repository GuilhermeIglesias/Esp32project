import cv2
import numpy as np
import requests

url = "http://192.168.137.207:81/stream"

stream = requests.get(url, stream=True)
bytes_data = bytes()

for chunk in stream.iter_content(chunk_size=1024):
    bytes_data += chunk
    a = bytes_data.find(b'\xff\xd8')
    b = bytes_data.find(b'\xff\xd9')

    if a != -1 and b != -1:
        jpg = bytes_data[a:b+2]
        bytes_data = bytes_data[b+2:]

        frame = cv2.imdecode(
            np.frombuffer(jpg, dtype=np.uint8),
            cv2.IMREAD_COLOR
        )

        cv2.imshow("ESP32 Stream", frame)

        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()
