import cv2
import numpy as np
from diagram_block import DiagramBlock
from text_parser import extract_swimlane_name


class Swimlane:
    """Класс для представления swimlane (плавательной дорожки)"""
    def __init__(self, id: int, y_top: int, y_bottom: int, x_left: int = 0, x_right: int = 0, name: str = ""):
        self.id = id
        self.y_top = y_top  # Верхняя граница полосы
        self.y_bottom = y_bottom  # Нижняя граница полосы
        self.x_left = x_left  # Левая граница полосы
        self.x_right = x_right  # Правая граница полосы
        self.name = name
    
    def __repr__(self):
        return f"Swimlane(id={self.id}, y_top={self.y_top}, y_bottom={self.y_bottom}, x_left={self.x_left}, x_right={self.x_right}, name='{self.name}')"


def process_swimlanes(image: np.ndarray, blocks: list[DiagramBlock], 
                      vertical_threshold: int = 30, 
                      text_search_width: int = 200) -> list[Swimlane]:
    """
    Обрабатывает swimline блоки: группирует их по высоте, определяет границы и названия.
    
    Args:
        image: Исходное изображение диаграммы
        blocks: Список всех блоков диаграммы
        vertical_threshold: Порог по Y для объединения swimline в одну группу
        text_search_width: Ширина области слева для поиска текста названия swimlane (в пикселях)
    
    Returns:
        Список swimlane с границами и названиями
    """
    # Фильтруем только swimline блоки
    swimline_blocks = [b for b in blocks if b.type == 'Swimline']
    
    if not swimline_blocks:
        return []
    
    # Группируем swimline по вертикальной позиции
    swimline_groups = _group_swimlines_by_height(swimline_blocks, vertical_threshold)
    
    # Создаем список swimlane с границами
    swimlanes = []
    _, image_width = image.shape[:2]
    
    for i, group in enumerate(swimline_groups):
        # Вычисляем верхнюю и нижнюю границы полосы
        y_coords_top = [b.bbox[1] for b in group]  # y1 из bbox - верх блока
        y_coords_bottom = [b.bbox[3] for b in group]  # y2 из bbox - низ блока
        y_top = min(y_coords_top)
        y_bottom = max(y_coords_bottom)
        
        # Расширяем swimlane на всю ширину изображения
        x_left = 0
        x_right = image_width
        
        # Определяем название swimlane из текста слева (используем только левую часть)
        name = extract_swimlane_name(image, group, text_search_width)
        
        swimlane = Swimlane(id=i, y_top=y_top, y_bottom=y_bottom, 
                           x_left=x_left, x_right=x_right, name=name)
        swimlanes.append(swimlane)
    
    # Сортируем swimlane по верхней границе
    swimlanes.sort(key=lambda s: s.y_top)
    
    # Переназначаем id после сортировки
    for i, swimlane in enumerate(swimlanes):
        swimlane.id = i
    
    # Присваиваем id swimlane всем блокам
    _assign_swimlane_to_blocks(blocks, swimlanes)
    
    return swimlanes


def _group_swimlines_by_height(swimline_blocks: list[DiagramBlock], 
                               threshold: int) -> list[list[DiagramBlock]]:
    """
    Группирует swimline блоки, которые находятся примерно на одной высоте.
    """
    if not swimline_blocks:
        return []
    
    # Сортируем по Y-координате (используем центр блока для более точной группировки)
    sorted_blocks = sorted(swimline_blocks, key=lambda b: (b.bbox[1] + b.bbox[3]) // 2)
    
    groups = []
    current_group = [sorted_blocks[0]]
    # Используем среднюю Y-координату первого блока как эталон для группы
    group_avg_y = (sorted_blocks[0].bbox[1] + sorted_blocks[0].bbox[3]) // 2
    
    for block in sorted_blocks[1:]:
        # Вычисляем центр текущего блока
        block_center_y = (block.bbox[1] + block.bbox[3]) // 2
        
        # Сравниваем со средним значением текущей группы
        y_diff = abs(block_center_y - group_avg_y)
        
        if y_diff <= threshold:
            # Добавляем в текущую группу
            current_group.append(block)
            # Обновляем среднее значение группы
            group_avg_y = sum((b.bbox[1] + b.bbox[3]) // 2 for b in current_group) // len(current_group)
        else:
            # Начинаем новую группу
            groups.append(current_group)
            current_group = [block]
            group_avg_y = block_center_y
    
    # Добавляем последнюю группу
    if current_group:
        groups.append(current_group)
    
    return groups


def _assign_swimlane_to_blocks(blocks: list[DiagramBlock], swimlanes: list[Swimlane]):
    """
    Присваивает каждому блоку id swimlane, в которой он находится.
    Блок принадлежит swimlane, если его центр находится между верхней и нижней границами полосы.
    """
    if not swimlanes:
        return
    
    for block in blocks:
        if block.type == 'Swimline':
            continue
        
        # Получаем центр блока по Y
        _, y1, _, y2 = block.bbox
        center_y = (y1 + y2) // 2
        
        # Находим swimlane, в которой находится блок (проверяем попадание в диапазон)
        for swimlane in swimlanes:
            if swimlane.y_top <= center_y <= swimlane.y_bottom:
                block.swimline = swimlane.id
                break
        else:
            # Если не попал ни в одну swimlane, присваиваем ближайшую
            if swimlanes:
                min_dist = float('inf')
                closest_swimlane = swimlanes[0]
                for swimlane in swimlanes:
                    # Вычисляем расстояние до границ swimlane
                    if center_y < swimlane.y_top:
                        dist = swimlane.y_top - center_y
                    elif center_y > swimlane.y_bottom:
                        dist = center_y - swimlane.y_bottom
                    else:
                        dist = 0
                    
                    if dist < min_dist:
                        min_dist = dist
                        closest_swimlane = swimlane
                block.swimline = closest_swimlane.id
