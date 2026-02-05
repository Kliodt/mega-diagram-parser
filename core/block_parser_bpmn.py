import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import torch

from diagram_block import DiagramBlock


def parse_blocks(image: np.ndarray) -> list[DiagramBlock]:
    trained_model = YOLO("./model/best.pt")

    # Запуск инференса на переданном изображении
    predictions = trained_model.predict(
        source=image,
        conf=0.25,  # threshold доверия
        device=0 if torch.cuda.is_available() else 'cpu',
        verbose=False
    )

    blocks = []
    
    # Обрабатываем результаты предсказания
    if predictions and len(predictions) > 0:
        result = predictions[0]  # Берем первый результат (наше изображение)
        
        # Извлекаем боксы и классы
        boxes = result.boxes
        
        for box in boxes:
            # Получаем координаты бокса (x1, y1, x2, y2)
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, xyxy)
            
            # Получаем класс объекта
            cls_id = int(box.cls[0].cpu().numpy())
            class_name = result.names[cls_id]
            
            # Создаем DiagramBlock
            block = DiagramBlock(type=class_name, bbox=(x1, y1, x2, y2))
            blocks.append(block)
    
    return blocks
