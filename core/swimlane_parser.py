import cv2
import numpy as np
from diagram_block import DiagramBlock
from text_parser import extract_swimlane_name


class Swimlane:
    """Класс для представления swimlane (плавательной дорожки)"""
    def __init__(self, id: int, y_boundary: int, name: str = ""):
        self.id = id
        self.y_boundary = y_boundary  # Граница по Y
        self.name = name
    
    def __repr__(self):
        return f"Swimlane(id={self.id}, y={self.y_boundary}, name='{self.name}')"


def process_swimlanes(image: np.ndarray, blocks: list[DiagramBlock], 
                      vertical_threshold: int = 30) -> list[Swimlane]:
    """
    Обрабатывает swimline блоки: группирует их по высоте, определяет границы и названия.
    
    Args:
        image: Исходное изображение диаграммы
        blocks: Список всех блоков диаграммы
        vertical_threshold: Порог по Y для объединения swimline в одну группу
    
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
    for i, group in enumerate(swimline_groups):
        # Вычисляем среднюю Y-координату границы
        y_coords = [b.bbox[1] for b in group]  # y1 из bbox
        y_boundary = int(np.mean(y_coords))
        
        # Определяем название swimlane из текста слева
        name = extract_swimlane_name(image, group)
        
        swimlane = Swimlane(id=i, y_boundary=y_boundary, name=name)
        swimlanes.append(swimlane)
    
    # Сортируем swimlane по Y-координате
    swimlanes.sort(key=lambda s: s.y_boundary)
    
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
    
    # Сортируем по Y-координате
    sorted_blocks = sorted(swimline_blocks, key=lambda b: b.bbox[1])
    
    groups = []
    current_group = [sorted_blocks[0]]
    
    for block in sorted_blocks[1:]:
        # Сравниваем с последним блоком в текущей группе
        last_block = current_group[-1]
        y_diff = abs(block.bbox[1] - last_block.bbox[1])
        
        if y_diff <= threshold:
            # Добавляем в текущую группу
            current_group.append(block)
        else:
            # Начинаем новую группу
            groups.append(current_group)
            current_group = [block]
    
    # Добавляем последнюю группу
    groups.append(current_group)
    
    return groups


def _assign_swimlane_to_blocks(blocks: list[DiagramBlock], swimlanes: list[Swimlane]):
    """
    Присваивает каждому блоку id swimlane, в которой он находится.
    """
    if not swimlanes:
        return
    
    # Создаем список границ swimlane
    boundaries = [s.y_boundary for s in swimlanes]
    boundaries.append(float('inf'))  # Добавляем верхнюю границу
    
    for block in blocks:
        if block.type == 'Swimline':
            continue
        
        # Получаем центр блока по Y
        _, y1, _, y2 = block.bbox
        center_y = (y1 + y2) // 2
        
        # Находим swimlane, в которой находится блок
        for i in range(len(swimlanes)):
            if i == 0:
                # Первая swimlane: от начала до первой границы
                if center_y < boundaries[i]:
                    block.swimline = i
                    break
            else:
                # Последующие swimlane: между границами
                if boundaries[i-1] <= center_y < boundaries[i]:
                    block.swimline = i
                    break
        else:
            # Если не попал ни в одну swimlane, присваиваем последнюю
            if swimlanes:
                block.swimline = len(swimlanes) - 1
