import socket
import struct
import cv2
import numpy as np

HOST = '0.0.0.0'
PORT = 5000

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((HOST, PORT))
server.listen(1)

print("Aguardando ESP32...")

while True:
    conn, addr = server.accept()
    print("Conectado por", addr)

    try:
        while True:
            data = conn.recv(4)
            if not data:
                break

            size = struct.unpack('<I', data)[0]

            img_data = b''
            while len(img_data) < size:
                packet = conn.recv(4096)
                if not packet:
                    break
                img_data += packet

            img = cv2.imdecode(
                np.frombuffer(img_data, np.uint8),
                cv2.IMREAD_COLOR
            )

            if img is not None:
                cv2.imshow("ESP32-CAM", img)

            if cv2.waitKey(1) == 27:
                raise KeyboardInterrupt

    except KeyboardInterrupt:
        conn.close()
        break

    conn.close()

server.close()
cv2.destroyAllWindows()