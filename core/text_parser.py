import pytesseract
import cv2
import numpy as np

from block_parser_bpmn import DiagramBlock


def set_tesseract_path(path):
     pytesseract.pytesseract.tesseract_cmd = path


def parse_inner_texts(image: np.ndarray, blocks: list[DiagramBlock]):
    for b in blocks:
        txt = _extract_text(image, b.bbox[0], b.bbox[1], b.bbox[2], b.bbox[3])
        b.inner_text = txt

def _extract_text(image: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> str:
        """
        Извлечение текста из области элемента с помощью OCR
        """
        # Добавляем отступ
        padding = 5
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(image.shape[1], x2 + padding)
        y2 = min(image.shape[0], y2 + padding)
        
        # Вырезаем область
        roi = image[y1:y2, x1:x2]
        
        if roi.size == 0:
            return ""
        
        # Предобработка для OCR
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi_thresh = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        # Увеличиваем для лучшего распознавания
        scale = 2
        roi_scaled = cv2.resize(roi_thresh, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        try:
            # OCR с настройками для русского и английского
            text = pytesseract.image_to_string(
                roi_scaled, 
                lang='rus+eng',
                config='--psm 6'  # Предполагаем единый блок текста
            )
            return text.strip().replace('\n', ' ')
        except Exception as e:
            print(f"Ошибка OCR: {e}")
            return ""


def extract_swimlane_name(image: np.ndarray, swimline_group: list) -> str:
    """
    Извлекает название swimlane из текста в левой части, повернутого на 90 градусов.
    
    Args:
        image: Исходное изображение диаграммы
        swimline_group: Список блоков swimline в одной группе
        
    Returns:
        Название swimlane
    """
    if not swimline_group:
        return ""
    
    # Находим самый левый блок в группе
    leftmost_block = min(swimline_group, key=lambda b: b.bbox[0])
    x1, y1, x2, y2 = leftmost_block.bbox
    
    # Увеличиваем область для захвата текста
    padding = 10
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(image.shape[1], x2 + padding)
    y2 = min(image.shape[0], y2 + padding)
    
    # Извлекаем ROI (область интереса)
    roi = image[y1:y2, x1:x2]
    
    if roi.size == 0:
        return ""
    
    # Поворачиваем изображение на 90 градусов по часовой стрелке
    # (текст был повернут на 90 против часовой, поэтому поворачиваем обратно)
    rotated = cv2.rotate(roi, cv2.ROTATE_90_CLOCKWISE)
    
    try:
        # OCR с настройками для русского и английского
        text = pytesseract.image_to_string(rotated, lang='rus+eng')
        return text.strip()
    except Exception as e:
        print(f"Ошибка OCR swimlane: {e}")
        return ""