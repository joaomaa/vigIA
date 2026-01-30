import cv2
import cvzone
import math
import time
import os
import numpy as np
import threading
import sqlite3  # <--- NOVO: Banco de Dados
from ultralytics import YOLO
from sort import Sort

# ==========================================
# 1. CONFIGURAÇÕES
# ==========================================
USERNAME = 'admin'
PASSWORD = '@Prodaterdpi1'
IP = '10.10.150.71'
URL = f"rtsp://{USERNAME}:{PASSWORD}@{IP}:554/onvif1"
DISPLAY_W, DISPLAY_H = 1280, 720

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|reorder_queue_size;500|buffer_size;1024000"
TEMPO_TOLERANCIA_SAIDA = 5.0 
LIMITE_FUNDO_Y = 650 

# ==========================================
# 2. FUNÇÕES DE BANCO DE DADOS (MICRODADOS)
# ==========================================
def init_db():
    conn = sqlite3.connect('vigia.db')
    cursor = conn.cursor()
    # Tabela simples para guardar os acessos
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS acessos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pessoa_id INTEGER,
            entrada DATETIME,
            saida DATETIME,
            duracao TEXT,
            foto_entrada TEXT,
            status TEXT
        )
    ''')
    conn.commit()
    conn.close()

def registrar_entrada_db(pessoa_id, foto_path):
    conn = sqlite3.connect('vigia.db')
    cursor = conn.cursor()
    hora_atual = time.strftime("%Y-%m-%d %H:%M:%S")
    
    cursor.execute('''
        INSERT INTO acessos (pessoa_id, entrada, foto_entrada, status)
        VALUES (?, ?, ?, 'Presente')
    ''', (pessoa_id, hora_atual, foto_path))
    
    conn.commit()
    conn.close()

def registrar_saida_db(pessoa_id, duracao_str):
    conn = sqlite3.connect('vigia.db')
    cursor = conn.cursor()
    hora_atual = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Atualiza o último registro deste ID que ainda esteja como 'Presente'
    cursor.execute('''
        UPDATE acessos 
        SET saida = ?, duracao = ?, status = 'Finalizado'
        WHERE pessoa_id = ? AND status = 'Presente'
    ''', (hora_atual, duracao_str, pessoa_id))
    
    conn.commit()
    conn.close()

# ==========================================
# 3. CLASSE DE VIDEO
# ==========================================
class VideoStream:
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        self.status, self.frame = self.capture.read()
        self.stop_thread = False
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while True:
            if self.stop_thread: return
            status, frame = self.capture.read()
            if status:
                self.frame = frame
                self.status = True
            else:
                self.status = False
                time.sleep(0.1)

    def get_frame(self):
        return self.status, self.frame

    def stop(self):
        self.stop_thread = True
        self.capture.release()

# ==========================================
# 4. INICIALIZAÇÃO
# ==========================================
init_db() # Cria o banco se não existir

cam = VideoStream(URL)
time.sleep(1.0)
model = YOLO("../Yolo-Weights/yolov8n.pt")
tracker = Sort(max_age=50, min_hits=3, iou_threshold=0.3)

people_data = {}

# Garante que a pasta static existe para o site ver as fotos depois
if not os.path.exists("static/fotos"): os.makedirs("static/fotos")

print("-" * 50)
print("SISTEMA VIGIA + BANCO DE DADOS")
print("Dados sendo salvos em 'vigia.db'")
print("-" * 50)

# ==========================================
# 5. LOOP PRINCIPAL
# ==========================================
while True:
    success, img_raw = cam.get_frame()
    if not success:
        time.sleep(0.1)
        continue

    img = cv2.resize(img_raw, (DISPLAY_W, DISPLAY_H))
    
    results = model(img, stream=True, verbose=False, conf=0.45)
    detections = np.empty((0, 5))

    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = math.ceil((box.conf[0] * 100)) / 100
                detections = np.vstack((detections, np.array([x1, y1, x2, y2, conf])))

    resultsTracker = tracker.update(detections)
    current_frame_ids = []

    for result in resultsTracker:
        x1, y1, x2, y2, id = map(int, result)
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w // 2, y1 + h 
        
        current_frame_ids.append(id)
        timestamp_agora = time.time()

        # --- LÓGICA DE ENTRADA ---
        if id not in people_data:
            people_data[id] = {
                'entrada': timestamp_agora,
                'ultimo_visto': timestamp_agora,
                'saiu': False,
                'status_loc': 'Sala'
            }
            
            # Salva Foto na pasta 'static/fotos' (para o site ler)
            hora_nome = time.strftime("%Y%m%d_%H%M%S")
            filename = f"foto_{id}_{hora_nome}.jpg"
            filepath = os.path.join("static/fotos", filename)
            cv2.imwrite(filepath, img)
            
            # GRAVA NO BANCO
            registrar_entrada_db(id, filename)
            print(f"[DB] ID {id} registrado no banco.")

        else:
            people_data[id]['ultimo_visto'] = timestamp_agora
            people_data[id]['saiu'] = False
            
            if cy > LIMITE_FUNDO_Y: people_data[id]['status_loc'] = 'Fundo'
            else: people_data[id]['status_loc'] = 'Sala'

        # Desenho
        status_txt = people_data[id]['status_loc']
        color = (0, 165, 255) if status_txt == 'Fundo' else (0, 255, 0)
        
        tempo_decorrido = int(timestamp_agora - people_data[id]['entrada'])
        mins, segs = divmod(tempo_decorrido, 60)
        
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=color)
        cvzone.putTextRect(img, f"ID:{id} {mins}m{segs}s", (x1, y1-10), scale=1, thickness=1, colorR=color)

    # --- LÓGICA DE SAÍDA ---
    agora = time.time()
    for person_id in list(people_data.keys()):
        person = people_data[person_id]
        if person['saiu']: continue
            
        if person_id not in current_frame_ids:
            tempo_sumido = agora - person['ultimo_visto']
            
            if tempo_sumido > TEMPO_TOLERANCIA_SAIDA:
                people_data[person_id]['saiu'] = True
                
                duracao_total = person['ultimo_visto'] - person['entrada']
                mins, segs = divmod(int(duracao_total), 60)
                duracao_str = f"{mins}m {segs}s"
                
                # ATUALIZA O BANCO COM A SAÍDA
                registrar_saida_db(person_id, duracao_str)
                print(f"[DB] ID {person_id} finalizado. Duração: {duracao_str}")

    cv2.imshow("Sistema Vigia (Gravando DB)", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cam.stop()
        break

cv2.destroyAllWindows()