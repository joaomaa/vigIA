from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

#cap = cv2.VideoCapture(0)
#cap.set(3, 1280)
#cap.set(4, 720)
cap = cv2.VideoCapture("../Videos/cars.mp4")

model = YOLO('../Yolo-Weights/yolov8n.pt')

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat"]

mask = cv2.imread("mask.png")

# Traking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
limits = [400, 297, 673, 297]
totalCount = []

while True:
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img, mask)
    imgGraphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img, imgGraphics, (0, 0))
    results = model(imgRegion, stream=True)


    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        
        # Para cada caixa detectada (objeto)
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Imprime as coordenadas no terminal
            print(x1, y1, x2, y2)
            
            # Desenha o retângulo na imagem
            # (imagem, ponto_inicial, ponto_final, cor_roxa, espessura)
            #cv2.rectangle(img, (x1,y1), (x2,y2), (255, 0, 255), 3)
            w, h = x2-x1, y2-y1
            cvzone.cornerRect(frame, (x1, y1, w, h))
            #--> Confidence 
            conf = math.ceil((box.conf[0] * 100)) / 100
            print(conf)

            #Class Name
            cls = int(box.cls[0])
            nome_obj = model.names[cls]
            cvzone.putTextRect(frame, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=3, thickness=5)
    
    if not ret:
        print("Erro no frame")
        break

    cv2.namedWindow('VIDEO', cv2.WINDOW_NORMAL)
    cv2.imshow('VIDEO', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        print("Encerrando o sistema")
        break

cap.release()
cv2.destroyAllWindows()