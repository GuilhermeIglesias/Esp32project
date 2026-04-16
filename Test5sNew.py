import cv2
import numpy as np
import requests
import threading
import time
import scipy.optimize
from ultralytics import YOLO

# ================================================================
# CONFIGURAÇÃO
# ================================================================
URL_right = "http://10.225.89.205"
URL_left  = "http://10.225.89.43"
BASELINE  = 15.0

TEMPO_ESTAVEL = 5.0
MOVIMENTO_MAX = 30
CONF_MINIMA   = 0.4

fl       = None
tantheta = None
# ================================================================

# ---- Status das câmeras ----
status_cam = {"L": "aguardando...", "R": "aguardando..."}
frames_recebidos = {"L": 0, "R": 0}
ultimo_frame_tempo = {"L": 0.0, "R": 0.0}

model = YOLO("yolo12n.pt")
print("[SISTEMA] Modelo YOLO carregado com sucesso!")

frame_left  = None
frame_right = None
lock_l = threading.Lock()
lock_r = threading.Lock()

calibrado = False
calib_fl  = None
calib_tan = None
objetos_estaveis = {}

def log(msg):
    agora = time.strftime("%H:%M:%S")
    print(f"[{agora}] {msg}")

def ler_stream(url, lado):
    global frame_left, frame_right
    nome = "ESQUERDA" if lado == "L" else "DIREITA"
    tentativa = 0

    while True:
        tentativa += 1
        status_cam[lado] = f"conectando... (tentativa {tentativa})"
        log(f"[CAM {nome}] Tentando conectar em {url} (tentativa {tentativa})")

        try:
            r = requests.get(url + "/1024x768.mjpeg", stream=True, timeout=10)
            status_cam[lado] = "conectada ✓"
            log(f"[CAM {nome}] Conectada com sucesso!")
            buf = b""
            frames_consecutivos_ok = 0

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

                            frames_recebidos[lado] += 1
                            ultimo_frame_tempo[lado] = time.time()
                            frames_consecutivos_ok += 1

                            # Loga a cada 30 frames para não poluir
                            if frames_consecutivos_ok % 30 == 0:
                                log(f"[CAM {nome}] {frames_recebidos[lado]} frames recebidos no total")
                    except Exception as e:
                        pass  # JPEG corrompido, ignora

        except requests.exceptions.ConnectTimeout:
            status_cam[lado] = "timeout de conexão"
            log(f"[CAM {nome}] ERRO: Timeout ao conectar em {url} — câmera offline ou IP errado?")
        except requests.exceptions.ConnectionError:
            status_cam[lado] = "sem conexão"
            log(f"[CAM {nome}] ERRO: Sem conexão — verifique se a câmera está ligada e no WiFi")
        except requests.exceptions.ReadTimeout:
            status_cam[lado] = "timeout de leitura"
            log(f"[CAM {nome}] ERRO: Stream parou de responder")
        except Exception as e:
            status_cam[lado] = f"erro: {str(e)[:40]}"
            log(f"[CAM {nome}] ERRO inesperado: {e}")

        log(f"[CAM {nome}] Aguardando 3s antes de reconectar...")
        time.sleep(3)

t_l = threading.Thread(target=ler_stream, args=(URL_left,  "L"), daemon=True)
t_r = threading.Thread(target=ler_stream, args=(URL_right, "R"), daemon=True)
t_l.start()
t_r.start()
log("[SISTEMA] Threads de stream iniciadas para ambas as câmeras")

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
    global calibrado, calib_fl, calib_tan
    if not hasattr(calibrar_automatico, "pontos"):
        calibrar_automatico.pontos = []

    if len(boxes_l) == 0 or len(boxes_r) == 0:
        log("[CALIB] AVISO: nenhum objeto visível nas câmeras para calibrar")
        return

    areas_l = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes_l]
    areas_r = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes_r]
    bl = boxes_l[np.argmax(areas_l)]
    br = boxes_r[np.argmax(areas_r)]
    disp = centro_x(bl) - centro_x(br)

    if disp <= 0:
        log(f"[CALIB] AVISO: disparidade negativa ({disp:.1f}px) — câmeras podem estar trocadas!")
        return

    calibrar_automatico.pontos.append((dist_cm, disp, sz1))
    log(f"[CALIB] Ponto {len(calibrar_automatico.pontos)}/2 guardado: {dist_cm}cm, disparidade={disp:.1f}px")

    if len(calibrar_automatico.pontos) >= 2:
        d1, disp1, sz = calibrar_automatico.pontos[0]
        d2, disp2, _  = calibrar_automatico.pontos[-1]

        if d1 == d2:
            log("[CALIB] ERRO: os dois pontos têm a mesma distância! Use distâncias diferentes.")
            return

        calib_fl  = d1 - disp1 * d2 / disp2
        calib_tan = (1 / (d2 - calib_fl)) * (BASELINE / 2) * sz / disp2
        calibrado = True

        est1 = (BASELINE/2) * sz * (1/calib_tan) / disp1 + calib_fl
        est2 = (BASELINE/2) * sz * (1/calib_tan) / disp2 + calib_fl

        log(f"[CALIB] ✓ CALIBRAÇÃO CONCLUÍDA!")
        log(f"[CALIB] fl={calib_fl:.4f}  tantheta={calib_tan:.4f}")
        log(f"[CALIB] Verificação: {d1}cm → estimado {est1:.1f}cm | {d2}cm → estimado {est2:.1f}cm")
        log(f"[CALIB] Para usar sem recalibrar, coloca no topo do script:")
        log(f"[CALIB]   fl = {calib_fl:.6f}")
        log(f"[CALIB]   tantheta = {calib_tan:.6f}")

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

    for k, (i, j) in enumerate(pares):
        cor   = CORES[k % len(CORES)]
        nome  = model.names[cls_l[i]]
        box_l = boxes_l[i]
        box_r = boxes_r[j]
        chave = f"{nome}_{k}"

        if chave not in objetos_estaveis:
            objetos_estaveis[chave] = {
                "inicio": agora,
                "box_anterior": box_l,
                "medido": False,
                "distancia": None
            }
        else:
            estado = objetos_estaveis[chave]
            if box_moveu(box_l, estado["box_anterior"]):
                if estado["medido"]:
                    log(f"[DETECÇÃO] '{nome}' moveu-se — reiniciando contagem")
                estado["inicio"]   = agora
                estado["medido"]   = False
                estado["distancia"] = None
            else:
                tempo_parado = agora - estado["inicio"]
                if not estado["medido"] and tempo_parado >= TEMPO_ESTAVEL:
                    dist = calcular_distancia(box_l, box_r, sz1)
                    estado["distancia"] = dist
                    estado["medido"]    = True

                    if dist:
                        log(f"[MEDIÇÃO] '{nome}' está a {dist:.1f}cm das câmeras")
                    else:
                        log(f"[CALIB NECESSÁRIA] '{nome}' ficou estável! Informe a distância real:")
                        try:
                            dist_real = float(input("  >> Distância real em cm: "))
                            calibrar_automatico(dist_real, boxes_l, boxes_r, sz1)
                        except ValueError:
                            log("[CALIB] Valor inválido digitado, ignorando")

            estado["box_anterior"] = box_l

        estado = objetos_estaveis[chave]
        if estado["distancia"]:
            txt = f"{nome}: {estado['distancia']:.1f}cm"
        elif not calibrado:
            tempo_parado = agora - estado["inicio"]
            restante = max(0, TEMPO_ESTAVEL - tempo_parado)
            txt = f"{nome} (calib em {restante:.0f}s)" if restante > 0 else f"{nome} (calibrando...)"
        else:
            txt = nome

        tempo_parado = agora - objetos_estaveis[chave]["inicio"]
        progresso = min(1.0, tempo_parado / TEMPO_ESTAVEL)

        for img, box in [(img_l, box_l), (img_r, box_r)]:
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), cor, 2)
            tw = len(txt) * 9
            cv2.rectangle(img, (x1, y1-22), (x1+tw, y1), cor, cv2.FILLED)
            cv2.putText(img, txt, (x1+2, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
            barra_w = int((x2-x1) * progresso)
            cv2.rectangle(img, (x1, y2+2), (x1+barra_w, y2+7), (0,255,100), cv2.FILLED)
            cv2.rectangle(img, (x1, y2+2), (x2,          y2+7), (100,100,100), 1)

    chaves_ativas = {f"{model.names[cls_l[i]]}_{k}" for k, (i, _) in enumerate(pares)}
    objetos_estaveis = {k: v for k, v in objetos_estaveis.items() if k in chaves_ativas}

    # Status no canto
    if calibrado:
        txt_status = "CALIBRADO"
        cor_status = (0, 220, 0)
    else:
        pts = len(getattr(calibrar_automatico, "pontos", []))
        txt_status = f"SEM CALIB ({pts}/2 pontos)"
        cor_status = (0, 120, 255)
    cv2.putText(img_l, txt_status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor_status, 2)

    # Status das câmeras no canto da imagem
    fps_l = frames_recebidos["L"]
    fps_r = frames_recebidos["R"]
    cv2.putText(img_l, f"CAM ESQ: {status_cam['L']}", (10, img_l.shape[0]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)
    cv2.putText(img_r, f"CAM DIR: {status_cam['R']}", (10, img_r.shape[0]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)

    return img_l, img_r

# ================================================================
log("[SISTEMA] Iniciando Sistema Estéreo ESP32-CAM + YOLO")
log(f"[SISTEMA] Câmera esquerda: {URL_left}")
log(f"[SISTEMA] Câmera direita:  {URL_right}")
log(f"[SISTEMA] Baseline: {BASELINE}cm | Tempo estável: {TEMPO_ESTAVEL}s")
log("[SISTEMA] Aguardando conexão das câmeras...")

if fl is not None and tantheta is not None:
    calib_fl  = fl
    calib_tan = tantheta
    calibrado = True
    log(f"[SISTEMA] Calibração pré-carregada: fl={fl}  tantheta={tantheta}")

ultimo_log_aguarda = 0

while True:
    with lock_l: fl_f = frame_left.copy()  if frame_left  is not None else None
    with lock_r: fr_f = frame_right.copy() if frame_right is not None else None

    agora = time.time()

    if fl_f is not None and fr_f is not None:
        img_l, img_r = processar_e_desenhar(fl_f, fr_f)
        combined = np.hstack([img_l, img_r])
        cv2.imshow("Stereo ESP32-CAM", combined)
    else:
        # Loga o que está faltando a cada 5 segundos
        if agora - ultimo_log_aguarda > 5:
            falta = []
            if fl_f is None:
                falta.append(f"esquerda ({status_cam['L']})")
            if fr_f is None:
                falta.append(f"direita ({status_cam['R']})")
            log(f"[SISTEMA] Aguardando câmera(s): {', '.join(falta)}")
            ultimo_log_aguarda = agora

        espera = np.zeros((300, 800, 3), dtype=np.uint8)
        msgs = [
            "Aguardando cameras...",
            f"ESQ: {status_cam['L']}",
            f"DIR: {status_cam['R']}",
        ]
        for idx, m in enumerate(msgs):
            cor = (0,220,0) if "conectada" in m else (0,150,255)
            cv2.putText(espera, m, (30, 80 + idx*60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor, 2)
        cv2.imshow("Stereo ESP32-CAM", espera)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        log("[SISTEMA] Encerrando por solicitação do usuário...")
        break

cv2.destroyAllWindows()
log("[SISTEMA] Sistema encerrado.")