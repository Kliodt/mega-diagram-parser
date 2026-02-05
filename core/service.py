import cv2
import numpy as np
import json
from typing import Dict, Any, List

from block_parser_any import parse_blocks_any
from arrow_parser import parse_arrows, visualize_connections
from block_parser_bpmn import parse_blocks
from text_parser import parse_inner_texts
from swimlane_parser import process_swimlanes, Swimlane
from diagram_block import DiagramBlock
from arrow_parser import DiagramArrow


class DiagramParsingService:
    """
    Сервис для парсинга диаграмм.
    Выполняет обнаружение блоков и связей на изображении.
    """
    
    def __init__(self, proximity_threshold: int = 30):
        """
        Args:
            proximity_threshold: Порог близости для определения связей между блоками
        """
        self.proximity_threshold = proximity_threshold
    
    def parse_diagram(self, image: np.ndarray):
        """
        Выполняет полный парсинг диаграммы.
        
        Args:
            image: Изображение диаграммы в формате BGR
            
        Returns:
            Словарь с результатами парсинга:
            - blocks: список обнаруженных блоков
            - arrows: список связей между блоками
        """
        # Проверяем, что изображение на белом фоне
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        
        # Если фон темный (среднее значение < 127), инвертируем изображение
        if mean_brightness < 127:
            image = cv2.bitwise_not(image)
        
        # Парсим блоки
        # blocks = parse_blocks_any(image)
        blocks = parse_blocks(image)
        
        # Обрабатываем swimlane
        swimlanes = process_swimlanes(image, blocks, vertical_threshold=30)
        
        # Фильтруем swimline блоки
        blocks_no_swimline = [b for b in blocks if b.type != 'Swimline']

        # Парсим тексты у task
        task_blocks = [b for b in blocks if b.type == 'Task']
        parse_inner_texts(image, task_blocks)
        
        # Парсим стрелки/связи
        arrows = parse_arrows(
            image, 
            blocks_no_swimline, 
            proximity_threshold=self.proximity_threshold
        )
        
        return blocks, arrows, swimlanes
    
    def convert_to_json(self, blocks: List[DiagramBlock], 
                       arrows: List[DiagramArrow], 
                       swimlanes: List[Swimlane]) -> Dict[str, Any]:
        """
        Конвертирует результаты парсинга в JSON формат.
        
        Args:
            blocks: Список блоков диаграммы
            arrows: Список связей между блоками
            swimlanes: Список swimlane
            
        Returns:
            Словарь с actors, nodes и edges
        """
        # Создаем маппинг типов блоков на node_type
        type_mapping = {
            'End': 'end',
            'Exclusive': 'exclusive',
            'Input_file': 'document',
            'Output_file': 'document',
            'Parallel': 'parallel',
            'Receive': 'receive_message',
            'Send': 'send_message',
            'Start': 'start',
            'Task': 'task',
            'Empty_event': 'start',
            'Decision': 'decision',
            'Swimline': 'swimline'
        }
        
        # Конвертируем swimlanes в actors
        actors = []
        for swimlane in swimlanes:
            actors.append({
                "id": swimlane.id,
                "name": swimlane.name if swimlane.name else f"Actor {swimlane.id}"
            })
        
        # Фильтруем блоки без Swimline
        blocks_no_swimline = [b for b in blocks if b.type != 'Swimline']
        
        # Конвертируем blocks в nodes
        nodes = []
        for idx, block in enumerate(blocks_no_swimline):
            # Определяем node_type на основе типа блока
            node_type = type_mapping.get(block.type, 'task')
            
            node = {
                "id": idx,
                "node_type": node_type,
                "text": block.inner_text if block.inner_text else "",
                "actor_id": block.swimline if block.swimline >= 0 else 0
            }
            nodes.append(node)
        
        # Конвертируем arrows в edges
        edges = []
        for arrow in arrows:
            # Проверяем, что индексы валидны
            if (0 <= arrow.from_box < len(blocks_no_swimline) and 
                0 <= arrow.to_box < len(blocks_no_swimline)):
                edges.append([arrow.from_box, arrow.to_box])
        
        result = {
            "actors": actors,
            "nodes": nodes,
            "edges": edges
        }
        
        return result
    

