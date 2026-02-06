import cv2
import numpy as np
from io import BytesIO
from typing import Optional


def convert_svg_to_image(svg_data: bytes, width: Optional[int] = None, height: Optional[int] = None) -> Optional[np.ndarray]:
    """
    Конвертирует SVG в растровое изображение (numpy array).
    
    Args:
        svg_data: Байты SVG файла
        width: Желаемая ширина выходного изображения (опционально)
        height: Желаемая высота выходного изображения (опционально)
    
    Returns:
        Изображение в формате BGR (numpy array) или None при ошибке
    """
    try:
        import cairosvg
        
        # Параметры конвертации
        kwargs = {}
        if width:
            kwargs['output_width'] = width
        if height:
            kwargs['output_height'] = height
        
        # Конвертируем SVG в PNG байты
        png_data = cairosvg.svg2png(bytestring=svg_data, **kwargs)
        
        # Конвертируем PNG байты в numpy array
        nparr = np.frombuffer(png_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        return image
    except ImportError:
        print("Ошибка: библиотека cairosvg не установлена. Установите: pip install cairosvg")
        return None
    except Exception as e:
        print(f"Ошибка конвертации SVG: {e}")
        return None


def is_svg_file(file_data: bytes) -> bool:
    """
    Проверяет, является ли файл SVG по его содержимому.
    
    Args:
        file_data: Первые байты файла
    
    Returns:
        True если файл SVG, иначе False
    """
    try:
        # Декодируем первые байты для проверки
        header = file_data[:100].decode('utf-8', errors='ignore').lower()
        return '<svg' in header or '<?xml' in header and 'svg' in header
    except Exception:
        return False
