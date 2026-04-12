import cv2
import numpy as np
import requests
import torchvision
import torchvision.transforms.functional as tvtf
import torch
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights
import scipy.optimize

# ---- CONFIGURAÇÃO ----
#url = "http://10.216.23.205/1024x768.mjpeg"
#url = "http://10.216.23.143/1024x768.mjpeg"

URL_left  = "http://10.216.23.205"   # IP da câmera esquerda
URL_right = "http://10.216.23.143"   # IP da câmera direita
BASELINE  = 10.5   # os teus 10.5cm medidos

DIST_1 = 40  # primeira distância de calibração em cm
DIST_2 = 60  # segunda distância de calibração em cm

weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=weights)
_ = model.eval()

def capturar_frame(url):
    """Captura um único frame do teu stream MJPEG"""
    r = requests.get(url + "/1024x768.mjpeg", stream=True, timeout=10)
    buf = b""
    for chunk in r.iter_content(chunk_size=4096):
        buf += chunk
        a = buf.find(b"\xff\xd8")
        b_end = buf.find(b"\xff\xd9")
        if a != -1 and b_end != -1 and b_end > a:
            jpg = buf[a:b_end+2]
            r.close()
            frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def preprocess(img):
    return tvtf.to_tensor(img).unsqueeze(0)

def detectar_objeto_principal(frame, score_thresh=0.4):
    """Retorna o bounding box do objeto com maior score"""
    with torch.no_grad():
        result = model(preprocess(frame))[0]
    mask = result["scores"] > score_thresh
    boxes = result["boxes"][mask].cpu().numpy()
    lbls  = result["labels"][mask].cpu().numpy()
    scores = result["scores"][mask].cpu().numpy()
    cats  = np.array(weights.meta["categories"])
    
    if len(boxes) == 0:
        return None, None
    
    # mostra o que foi detetado
    for i, (b, l, s) in enumerate(zip(boxes, lbls, scores)):
        print(f"  [{i}] {cats[l]} ({s:.2f})")
    
    # devolve o de maior score
    best = np.argmax(scores)
    return boxes[best], cats[lbls[best]]

def centro_x(box):
    return (box[0] + box[2]) / 2

# ---- CALIBRAÇÃO DISTÂNCIA 1 ----
print(f"=== CALIBRAÇÃO ===")
print(f"Coloca o objeto a EXATAMENTE {DIST_1}cm das câmeras e aperte ENTER")
input()

frame_l1 = capturar_frame(URL_left)
frame_r1 = capturar_frame(URL_right)

cv2.imshow(f"Left {DIST_1}cm", cv2.cvtColor(frame_l1, cv2.COLOR_RGB2BGR))
cv2.imshow(f"Right {DIST_1}cm", cv2.cvtColor(frame_r1, cv2.COLOR_RGB2BGR))
cv2.waitKey(2000)

print(f"\nObjetos detetados na câmera ESQUERDA a {DIST_1}cm:")
box_l1, label_l1 = detectar_objeto_principal(frame_l1)
print(f"Objetos detetados na câmera DIREITA a {DIST_1}cm:")
box_r1, label_r1 = detectar_objeto_principal(frame_r1)

if box_l1 is None or box_r1 is None:
    print("ERRO: objeto não detetado! Tenta com score_thresh mais baixo ou objeto diferente.")
    exit()

disp_1 = centro_x(box_l1) - centro_x(box_r1)
print(f"\nDisparidade a {DIST_1}cm: {disp_1:.2f} píxeis  (deve ser positivo!)")

if disp_1 <= 0:
    print("AVISO: disparidade negativa! As câmeras podem estar trocadas (esquerda/direita).")
    print("Troca os URLs de URL_left e URL_right e volta a correr.")

# ---- CALIBRAÇÃO DISTÂNCIA 2 ----
print(f"\nColoca o objeto a EXATAMENTE {DIST_2}cm das câmeras e prime ENTER")
input()

frame_l2 = capturar_frame(URL_left)
frame_r2 = capturar_frame(URL_right)

cv2.imshow(f"Left {DIST_2}cm", cv2.cvtColor(frame_l2, cv2.COLOR_RGB2BGR))
cv2.imshow(f"Right {DIST_2}cm", cv2.cvtColor(frame_r2, cv2.COLOR_RGB2BGR))
cv2.waitKey(2000)

print(f"\nObjetos detetados na câmera ESQUERDA a {DIST_2}cm:")
box_l2, label_l2 = detectar_objeto_principal(frame_l2)
print(f"Objetos detetados na câmera DIREITA a {DIST_2}cm:")
box_r2, label_r2 = detectar_objeto_principal(frame_r2)

if box_l2 is None or box_r2 is None:
    print("ERRO: objeto não detetado!")
    exit()

disp_2 = centro_x(box_l2) - centro_x(box_r2)
print(f"Disparidade a {DIST_2}cm: {disp_2:.2f} píxeis")

# ---- CÁLCULO DOS PARÂMETROS ----
sz1 = frame_r1.shape[1]

fl = DIST_1 - disp_1 * DIST_2 / disp_2
tantheta = (1 / (DIST_2 - fl)) * (BASELINE / 2) * sz1 / disp_2

print(f"\n=============================")
print(f"  PARÂMETROS CALIBRADOS")
print(f"=============================")
print(f"  baseline  = {BASELINE} cm")
print(f"  fl        = {fl:.6f}")
print(f"  tantheta  = {tantheta:.6f}")
print(f"  sz1       = {sz1} píxeis")
print(f"=============================")
print(f"\nVerificação:")
dist_check_1 = (BASELINE/2) * sz1 * (1/tantheta) / disp_1 + fl
dist_check_2 = (BASELINE/2) * sz1 * (1/tantheta) / disp_2 + fl
print(f"  A {DIST_1}cm → modelo estima: {dist_check_1:.1f}cm (erro: {abs(dist_check_1-DIST_1):.1f}cm)")
print(f"  A {DIST_2}cm → modelo estima: {dist_check_2:.1f}cm (erro: {abs(dist_check_2-DIST_2):.1f}cm)")
print(f"\nCopia estes valores para o script principal!")

cv2.destroyAllWindows()