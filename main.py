import cv2
import cv2.aruco as aruco
import numpy as np
import time
import math
import random
import pygame

# =========================
# CONFIG
# =========================
cap = cv2.VideoCapture(0)

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
params = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, params)

# =========================
# AUDIO (adicione os arquivos na mesma pasta)
# =========================
pygame.mixer.init()

SOM_BATIDA = pygame.mixer.Sound("Som_da_batida.mp3")     # som ao bater no bastão
SOM_GOL = pygame.mixer.Sound("Som_de_ponto.mp3")       # som ao marcar ponto
SOM_VITORIA = pygame.mixer.Sound("som_de_vitoria.mp3")  # som de plateia / vitória

SOM_BATIDA.set_volume(0.5)
SOM_GOL.set_volume(0.7)
SOM_VITORIA.set_volume(0.9)

# IDs dos jogadores
ID_ESQ = 88
ID_DIR = 973

# Jogo
placar_esq = 0
placar_dir = 0
PONTOS_VITORIA = 5

# Jogadores
jogador_esq = None
jogador_dir = None
RAIO_JOGADOR = 35

# Puck
puck = None
RAIO_PUCK = 14
VEL_PUCK = 420

# Tempo
last_time = time.time()

# Vitória
vencedor = None
tempo_vitoria = 0


# =========================
# FUNÇÕES
# =========================
def criar_jogador(cx, cy, ang):
    return {
        "x": cx,
        "y": cy,
        "ang": ang,
        "last_x": cx,
        "last_y": cy,
        "vx": 0,
        "vy": 0
    }


def atualizar_jogador(jogador, cx, cy, ang, dt):
    if jogador is None:
        return criar_jogador(cx, cy, ang)

    jogador["vx"] = (cx - jogador["x"]) / max(dt, 1e-4)
    jogador["vy"] = (cy - jogador["y"]) / max(dt, 1e-4)

    jogador["last_x"] = jogador["x"]
    jogador["last_y"] = jogador["y"]

    jogador["x"] = cx
    jogador["y"] = cy
    jogador["ang"] = ang

    return jogador


def criar_puck(w, h, direcao=None):
    if direcao is None:
        direcao = random.choice([-1, 1])

    ang = random.uniform(-0.6, 0.6)

    return {
        "x": w // 2,
        "y": h // 2,
        "vx": direcao * VEL_PUCK * math.cos(ang),
        "vy": VEL_PUCK * math.sin(ang),
        "r": RAIO_PUCK
    }


def desenhar_jogador(frame, jogador, cor):
    x = int(jogador["x"])
    y = int(jogador["y"])
    ang = jogador["ang"]

    cv2.circle(frame, (x, y), RAIO_JOGADOR, cor, -1)
    cv2.circle(frame, (x, y), RAIO_JOGADOR, (255, 255, 255), 2)

    dx = int(x + math.cos(ang) * 28)
    dy = int(y - math.sin(ang) * 28)
    cv2.line(frame, (x, y), (dx, dy), (255, 255, 255), 3)

    cv2.circle(frame, (x, y), 7, (255, 255, 255), -1)


def desenhar_puck(frame, puck):
    cv2.circle(frame, (int(puck["x"]), int(puck["y"])), puck["r"], (255, 255, 255), -1)
    cv2.circle(frame, (int(puck["x"]), int(puck["y"])), puck["r"], (0, 0, 0), 2)


def desenhar_campo(frame, w, h):
    frame[:] = (30, 120, 30)

    cv2.line(frame, (w // 2, 0), (w // 2, h), (255, 255, 255), 2)
    cv2.circle(frame, (w // 2, h // 2), 80, (255, 255, 255), 2)

    gol_h = 180
    y1 = h // 2 - gol_h // 2
    y2 = h // 2 + gol_h // 2

    cv2.line(frame, (0, 0), (0, y1), (255, 255, 255), 6)
    cv2.line(frame, (0, y2), (0, h), (255, 255, 255), 6)

    cv2.line(frame, (w, 0), (w, y1), (255, 255, 255), 6)
    cv2.line(frame, (w, y2), (w, h), (255, 255, 255), 6)

    cv2.line(frame, (0, 0), (w, 0), (255, 255, 255), 6)
    cv2.line(frame, (0, h), (w, h), (255, 255, 255), 6)

    return y1, y2


def colidir_puck_jogador(puck, jogador):
    dx = puck["x"] - jogador["x"]
    dy = puck["y"] - jogador["y"]
    dist = math.sqrt(dx * dx + dy * dy)

    min_dist = puck["r"] + RAIO_JOGADOR

    if dist < min_dist and dist > 0:
        nx = dx / dist
        ny = dy / dist

        overlap = min_dist - dist
        puck["x"] += nx * overlap
        puck["y"] += ny * overlap

        dot = puck["vx"] * nx + puck["vy"] * ny
        puck["vx"] -= 2 * dot * nx
        puck["vy"] -= 2 * dot * ny

        puck["vx"] += jogador["vx"] * 0.35
        puck["vy"] += jogador["vy"] * 0.35
        puck["vy"] += -math.sin(jogador["ang"]) * 80

        speed = math.sqrt(puck["vx"]**2 + puck["vy"]**2)
        max_speed = 900

        if speed > max_speed:
            scale = max_speed / speed
            puck["vx"] *= scale
            puck["vy"] *= scale

        # SOM DE BATIDA
        SOM_BATIDA.play()


# =========================
# LOOP PRINCIPAL
# =========================
while True:
    ret, camera = cap.read()
    if not ret:
        break

    

    h, w, _ = camera.shape
    now = time.time()
    dt = now - last_time
    last_time = now

    frame = camera.copy()
    overlay = camera.copy()
    gol_y1, gol_y2 = desenhar_campo(overlay, w, h)

    # campo translúcido
    frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

    gray = cv2.cvtColor(camera, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    camera = cv2.flip(camera, 1)

    # -------------------------
    # DETECÇÃO DOS ARUCOS
    # -------------------------
    if ids is not None:
        for i in range(len(ids)):
            marker_id = ids[i][0]
            pts = corners[i][0]

            cx = int(pts[:, 0].mean())
            cy = int(pts[:, 1].mean())

            topo = pts[0]
            dx = topo[0] - cx
            dy = cy - topo[1]
            angulo = math.atan2(dy, dx)

            if marker_id == ID_ESQ:
                cx = min(cx, w // 2 - 30)
                jogador_esq = atualizar_jogador(jogador_esq, cx, cy, angulo, dt)

            elif marker_id == ID_DIR:
                cx = max(cx, w // 2 + 30)
                jogador_dir = atualizar_jogador(jogador_dir, cx, cy, angulo, dt)

    # -------------------------
    # CRIAR PUCK
    # -------------------------
    if puck is None:
        puck = criar_puck(w, h)

    # -------------------------
    # ATUALIZAR PUCK
    # -------------------------
    puck["x"] += puck["vx"] * dt
    puck["y"] += puck["vy"] * dt

    if puck["y"] - puck["r"] <= 0:
        puck["y"] = puck["r"]
        puck["vy"] *= -1

    if puck["y"] + puck["r"] >= h:
        puck["y"] = h - puck["r"]
        puck["vy"] *= -1

    if puck["x"] - puck["r"] <= 0:
        if not (gol_y1 <= puck["y"] <= gol_y2):
            puck["x"] = puck["r"]
            puck["vx"] *= -1

    if puck["x"] + puck["r"] >= w:
        if not (gol_y1 <= puck["y"] <= gol_y2):
            puck["x"] = w - puck["r"]
            puck["vx"] *= -1

    if jogador_esq is not None:
        colidir_puck_jogador(puck, jogador_esq)

    if jogador_dir is not None:
        colidir_puck_jogador(puck, jogador_dir)

    # -------------------------
    # GOL
    # -------------------------
    if puck["x"] < -20:
        placar_dir += 1
        SOM_GOL.play()
        puck = criar_puck(w, h, direcao=-1)

    elif puck["x"] > w + 20:
        placar_esq += 1
        SOM_GOL.play()
        puck = criar_puck(w, h, direcao=1)

    # Vitória
    if placar_esq >= PONTOS_VITORIA:
        vencedor = "VERMELHA VENCEU!"
        tempo_vitoria = time.time()
        SOM_VITORIA.play()
        placar_esq = 0
        placar_dir = 0
        puck = criar_puck(w, h)

    elif placar_dir >= PONTOS_VITORIA:
        vencedor = "AZUL VENCEU!"
        tempo_vitoria = time.time()
        SOM_VITORIA.play()
        placar_esq = 0
        placar_dir = 0
        puck = criar_puck(w, h)

    # -------------------------
    # DESENHO
    # -------------------------
    
    

    if jogador_esq is not None:
        desenhar_jogador(frame, jogador_esq, (0, 0, 255))

    if jogador_dir is not None:
        desenhar_jogador(frame, jogador_dir, (255, 0, 0))

    desenhar_puck(frame, puck)

    frame = cv2.flip(frame, 1)

    cv2.putText(frame, f"{placar_esq}", (w // 2 - 80, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

    cv2.putText(frame, f"{placar_dir}", (w // 2 + 40, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

    if vencedor and time.time() - tempo_vitoria < 2:
        cv2.putText(frame, vencedor, (w // 2 - 180, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
        
    
    cv2.imshow("Aruco Air Hockey", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
    elif key == ord('r'):
        placar_esq = 0
        placar_dir = 0
        puck = criar_puck(w, h)
        vencedor = None

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()