import cv2
import numpy as np
import requests
import threading
import time
import scipy.optimize
from ultralytics import YOLO


URL_right  = "http://10.225.89.205"   
URL_left = "http://10.225.89.43"  
BASELINE  = 15   # distância entre câmeras em cm

TEMPO_ESTAVEL = 5.0      # segundos parado para medir
MOVIMENTO_MAX = 30       # píxeis de tolerância para considerar parado
CONF_MINIMA   = 0.4      # confiança mínima YOLO

# Após calibração, coloca aqui os valores (ou deixa None para calibrar automaticamente)
fl       = None
tantheta = None

model = YOLO("yolo12n.pt")

frame_left  = None
frame_right = None
lock_l = threading.Lock()
lock_r = threading.Lock()

# Estado de calibração
calibrado = False
calib_fl  = None
calib_tan = None

# Rastreamento de estabilidade por objeto
objetos_estaveis = {}   # chave: nome_classe, valor: {tempo_inicio, box_anterior, distancia}

def ler_stream(url, lado):
    global frame_left, frame_right
    while True:
        try:
            r = requests.get(url + "/1024x768.mjpeg", stream=True, timeout=10)
            buf = b""
            for chunk in r.iter_content(chunk_size=4096):
                buf += chunk
                while True:
                    a     = buf.find(b"\xff\xd8")
                    b_end = buf.find(b"\xff\xd9")
                    if a == -1 or b_end == -1 or b_end <= a:
                        break
                    jpg = buf[a:b_end+2]
                    buf = buf[b_end+2:]
                    try:
                        frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                        if frame is not None:
                            if lado == "L":
                                with lock_l: frame_left = frame.copy()
                            else:
                                with lock_r: frame_right = frame.copy()
                    except:
                        pass  # ignora JPEGs corrompidos silenciosamente
        except:
            time.sleep(1)

t_l = threading.Thread(target=ler_stream, args=(URL_left,  "L"), daemon=True)
t_r = threading.Thread(target=ler_stream, args=(URL_right, "R"), daemon=True)
t_l.start()
t_r.start()

def capturar_frame_unico(url):
    try:
        r = requests.get(url + "/1024x768.mjpeg", stream=True, timeout=5)
        buf = b""
        for chunk in r.iter_content(chunk_size=4096):
            buf += chunk
            a     = buf.find(b"\xff\xd8")
            b_end = buf.find(b"\xff\xd9")
            if a != -1 and b_end != -1 and b_end > a:
                jpg = buf[a:b_end+2]
                r.close()
                return cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    except:
        pass
    return None

def detectar(frame, conf=CONF_MINIMA):
    results = model(frame, conf=conf, verbose=False)[0]
    boxes = results.boxes
    if boxes is None or len(boxes) == 0:
        return np.array([]), np.array([])
    return boxes.xyxy.cpu().numpy(), boxes.cls.cpu().numpy().astype(int)

def centro_x(box):
    return (box[0] + box[2]) / 2

def box_moveu(box_nova, box_antiga, limite=MOVIMENTO_MAX):
    if box_antiga is None:
        return True
    dx = abs(centro_x(box_nova) - centro_x(box_antiga))
    dy = abs((box_nova[1]+box_nova[3])/2 - (box_antiga[1]+box_antiga[3])/2)
    return dx > limite or dy > limite

def calcular_distancia(box_l, box_r, sz1):
    if calib_fl is None or calib_tan is None:
        return None
    disp = centro_x(box_l) - centro_x(box_r)
    if disp > 1:
        return (BASELINE / 2) * sz1 * (1 / calib_tan) / disp + calib_fl
    return None

def calibrar_automatico(dist_cm, boxes_l, boxes_r, sz1):
    """Guarda medição para calibração com dois pontos"""
    global calib_pontos
    if not hasattr(calibrar_automatico, "pontos"):
        calibrar_automatico.pontos = []

    # Usa o maior objeto de cada câmera
    if len(boxes_l) == 0 or len(boxes_r) == 0:
        return

    areas_l = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes_l]
    areas_r = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes_r]
    bl = boxes_l[np.argmax(areas_l)]
    br = boxes_r[np.argmax(areas_r)]
    disp = centro_x(bl) - centro_x(br)

    if disp <= 0:
        return

    calibrar_automatico.pontos.append((dist_cm, disp, sz1))
    print(f"[CALIB] Ponto guardado: {dist_cm}cm, disparidade={disp:.1f}px")

    if len(calibrar_automatico.pontos) >= 2:
        aplicar_calibracao()

def aplicar_calibracao():
    global calibrado, calib_fl, calib_tan
    pontos = calibrar_automatico.pontos

    # Usa os dois pontos mais distantes entre si
    d1, disp1, sz1 = pontos[0]
    d2, disp2, _   = pontos[-1]

    if d1 == d2:
        return

    calib_fl  = d1 - disp1 * d2 / disp2
    calib_tan = (1 / (d2 - calib_fl)) * (BASELINE / 2) * sz1 / disp2
    calibrado = True

    est1 = (BASELINE/2) * sz1 * (1/calib_tan) / disp1 + calib_fl
    est2 = (BASELINE/2) * sz1 * (1/calib_tan) / disp2 + calib_fl

    print(f"\n✓ CALIBRAÇÃO CONCLUÍDA!")
    print(f"  fl={calib_fl:.4f}  tantheta={calib_tan:.4f}")
    print(f"  Verificação: {d1}cm→{est1:.1f}cm | {d2}cm→{est2:.1f}cm\n")

def processar_e_desenhar(frame_l, frame_r):
    global objetos_estaveis

    boxes_l, cls_l = detectar(frame_l)
    boxes_r, cls_r = detectar(frame_r)
    sz1 = frame_r.shape[1]

    img_l = frame_l.copy()
    img_r = frame_r.copy()

    agora = time.time()
    CORES = [
        (255, 80, 80), (80, 255, 80), (80, 80, 255),
        (255, 220, 50), (50, 220, 255), (220, 50, 255)
    ]

    # Emparelha objetos entre câmera esquerda e direita
    pares = []
    if len(boxes_l) > 0 and len(boxes_r) > 0:
        n_l, n_r = len(boxes_l), len(boxes_r)
        cost = np.full((n_l, n_r), 99999.0)
        for i in range(n_l):
            for j in range(n_r):
                if cls_l[i] != cls_r[j]:
                    continue
                dy = abs((boxes_l[i][1]+boxes_l[i][3])/2 - (boxes_r[j][1]+boxes_r[j][3])/2)
                dx = centro_x(boxes_l[i]) - centro_x(boxes_r[j])
                if dx < 0: dx = 10 * abs(dx)
                cost[i, j] = 5*dy + dx

        tracks = scipy.optimize.linear_sum_assignment(cost)
        for i, j in zip(*tracks):
            if cost[i, j] < 99999:
                pares.append((i, j))

    # Processa cada par
    for k, (i, j) in enumerate(pares):
        cor   = CORES[k % len(CORES)]
        nome  = model.names[cls_l[i]]
        box_l = boxes_l[i]
        box_r = boxes_r[j]

        # Verifica estabilidade
        chave = f"{nome}_{k}"
        if chave not in objetos_estaveis:
            objetos_estaveis[chave] = {
                "inicio": agora,
                "box_anterior": box_l,
                "medido": False,
                "distancia": None,
                "dist_calib": None
            }
        else:
            estado = objetos_estaveis[chave]
            if box_moveu(box_l, estado["box_anterior"]):
                # Objeto moveu — reinicia contagem
                estado["inicio"] = agora
                estado["medido"] = False
                estado["distancia"] = None
            else:
                tempo_parado = agora - estado["inicio"]

                if not estado["medido"] and tempo_parado >= TEMPO_ESTAVEL:
                    # Objeto estável — mede distância
                    dist = calcular_distancia(box_l, box_r, sz1)
                    estado["distancia"] = dist
                    estado["medido"] = True

                    if dist:
                        print(f"[MEDIÇÃO] {nome}: {dist:.1f}cm")
                    else:
                        # Sem calibração — pede distância real ao utilizador
                        print(f"\n[CALIB AUTOMÁTICA] '{nome}' estável!")
                        try:
                            dist_real = float(input(f"  Qual a distância real em cm? "))
                            estado["dist_calib"] = dist_real
                            calibrar_automatico(dist_real, boxes_l, boxes_r, sz1)
                        except:
                            pass

            estado["box_anterior"] = box_l

        # Prepara texto para mostrar
        estado = objetos_estaveis[chave]
        if estado["distancia"]:
            txt = f"{nome}: {estado['distancia']:.1f}cm"
        elif not calibrado:
            tempo_parado = agora - estado["inicio"]
            restante = max(0, TEMPO_ESTAVEL - tempo_parado)
            if restante > 0:
                txt = f"{nome} (calib em {restante:.0f}s)"
            else:
                txt = f"{nome} (a calibrar...)"
        else:
            txt = nome

        # Desenha barra de progresso de estabilidade
        tempo_parado = agora - objetos_estaveis[chave]["inicio"]
        progresso = min(1.0, tempo_parado / TEMPO_ESTAVEL)

        for img, box in [(img_l, box_l), (img_r, box_r)]:
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), cor, 2)

            # Label com distância
            tw = len(txt) * 9
            cv2.rectangle(img, (x1, y1-22), (x1+tw, y1), cor, cv2.FILLED)
            cv2.putText(img, txt, (x1+2, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)

            # Barra de progresso de estabilidade (verde vai crescendo)
            barra_w = int((x2-x1) * progresso)
            cv2.rectangle(img, (x1, y2+2), (x1+barra_w, y2+7), (0,255,100), cv2.FILLED)
            cv2.rectangle(img, (x1, y2+2), (x2,          y2+7), (100,100,100), 1)

    # Remove objetos que desapareceram
    chaves_ativas = {f"{model.names[cls_l[i]]}_{k}" for k, (i, _) in enumerate(pares)}
    objetos_estaveis = {k: v for k, v in objetos_estaveis.items() if k in chaves_ativas}

    # Estado no canto superior esquerdo
    if calibrado:
        estado_txt = "CALIBRADO"
        cor_estado = (0, 220, 0)
    else:
        pts = len(getattr(calibrar_automatico, "pontos", []))
        estado_txt = f"SEM CALIBRACAO ({pts}/2 pontos)"
        cor_estado = (0, 120, 255)

    cv2.putText(img_l, estado_txt, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor_estado, 2)

    return img_l, img_r


print("Sistema Stereo ESP32-CAM + YOLO")
print("O objeto deve ficar 5s parado para ser medido/calibrado")
print("Prima Q na janela para sair")
print("=" * 45)

# Se tver fl e tantheta de sessões anteriores, ativar aqui:
if fl is not None and tantheta is not None:
    calib_fl  = fl
    calib_tan = tantheta
    calibrado = True
    print(f"Calibração carregada: fl={fl}  tantheta={tantheta}")

while True:
    with lock_l: fl_f = frame_left.copy()  if frame_left  is not None else None
    with lock_r: fr_f = frame_right.copy() if frame_right is not None else None

    if fl_f is not None and fr_f is not None:
        img_l, img_r = processar_e_desenhar(fl_f, fr_f)
        combined = np.hstack([img_l, img_r])
        cv2.imshow("Stereo ESP32-CAM", combined)
    else:
        # Mostra mensagem enquanto espera as câmeras
        espera = np.zeros((300, 700, 3), dtype=np.uint8)
        cv2.putText(espera, "A aguardar cameras...", (50, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)
        cv2.imshow("Stereo ESP32-CAM", espera)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
