import cv2
import numpy as np
from collections import deque

from diagram_block import DiagramBlock


def parse_blocks_any(image: np.ndarray) -> list[DiagramBlock]:

    # Преобразуем в серую шкалу
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Применяем пороговую обработку для бинаризации
    # _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # Находим контуры
    contours, _ = cv2.findContours(
        binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Убираем слишком маленькие контуры
    contours = _filtered_contours_by_size(contours)

    # Исправить выделение фигур
    preprocessed_contours_tmp = []
    for contour in contours:
        pp_contours = _preprocess_contour(contour, min_distance=20)
        preprocessed_contours_tmp.extend(pp_contours)
    contours = preprocessed_contours_tmp

    # Снова убираем слишком маленькие контуры
    contours = _filtered_contours_by_size(contours, min_area=100, min_perimeter=50)
    contours = _filtered_contours_by_bbox_size(contours, min_bbox_wh=10)

    # Убираем все невыпуклые контуры
    contours = _filter_convex_contours(contours, convexity_threshold=0.9)

    # Убираем сдвоенные контуры
    contours = _remove_duplicate_contours(contours, iou_threshold=0.6)

    # Убираем контуры, которые не заполняют свой bbox
    # contours = _filter_contours_by_fill_ratio(contours, min_fill_ratio=0.5)


    # Выводим только отфильтрованные контуры
    # image_with_boxes = image.copy()
    # for i, contour in enumerate(contours):
    #     # Получаем bounding box для каждого контура
    #     x, y, w, h = cv2.boundingRect(contour)

    #     # Выводим координаты и форму
    #     print(x,y,x+w,y+h)

    #     # Рисуем сам контур (пиксели контура)
    #     # cv2.drawContours(image_with_boxes, [contour], 0, (255,255,0), 2)
    #     cv2.rectangle(image_with_boxes, [x,y], [x+w,y+h], (255,255,0), 2)

    #     # Выводим название формы и номер контура
    #     x, y, w, h = cv2.boundingRect(contour)
    #     label = f"{i + 1}."
    #     cv2.putText(image_with_boxes, label, (x, y - 10),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)

    # # Отображаем исходное и обработанное изображения
    # cv2.imshow('Original Image', image)
    # cv2.imshow('Contours with Bounding Boxes', image_with_boxes)
    # cv2.waitKey(-1)
    # cv2.destroyAllWindows()

    # Преобразуем контуры в DiagramBlock объекты
    diagram_blocks = []
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        block = DiagramBlock(bbox=(x,y,x+w,y+h))
        diagram_blocks.append(block)
    
    return diagram_blocks




def _filtered_contours_by_size(contours, min_area=100, min_perimeter=50):
    filtered = []
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        if perimeter < min_perimeter:
            continue
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        filtered.append(contour)
    return filtered


def _filtered_contours_by_bbox_size(contours, min_bbox_wh=10):
    filtered = []
    for contour in contours:
        _, _, w, h = cv2.boundingRect(contour)
        if w < min_bbox_wh or h < min_bbox_wh:
            continue
        filtered.append(contour)
    return filtered


def _filter_contours_by_fill_ratio(contours, min_fill_ratio=0.5, max_fill_ratio=1.0):
    """
    Фильтрует контуры по отношению их площади к площади bounding box.
    
    Args:
        contours: список контуров
        min_fill_ratio: минимальное отношение (площадь контура / площадь bbox).
                       Значение от 0 до 1. Например, 0.5 означает что контур 
                       должен занимать минимум 50% своего bounding box.
        max_fill_ratio: максимальное отношение (обычно 1.0 для всех контуров)
    
    Returns:
        Список контуров, удовлетворяющих условию по fill ratio
    """
    filtered = []
    
    for contour in contours:
        # Вычисляем площадь контура
        contour_area = cv2.contourArea(contour)
        
        # Вычисляем площадь bounding box
        _, _, w, h = cv2.boundingRect(contour)
        bbox_area = w * h
        
        if bbox_area == 0:
            continue
        
        # Вычисляем ratio (коэффициент заполнения)
        fill_ratio = contour_area / bbox_area
        
        # Проверяем условие
        if min_fill_ratio <= fill_ratio <= max_fill_ratio:
            filtered.append(contour)
    
    return filtered


def _remove_duplicate_contours(contours, iou_threshold=0.6):
    """
    Удаляет дублирующиеся контуры, оставляя только один (самый большой по площади).

    Args:
        contours: список контуров
        iou_threshold: IoU (Intersection over Union) = Площадь \
            пересечения (intersection) / площадь объединения (union)

    Returns:
        Список контуров с удалёнными дубликатами
    """
    if len(contours) == 0:
        return []

    # Вычисляем bounding boxes и площади
    bboxes = []
    areas = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bboxes.append((x, y, w, h))
        areas.append(cv2.contourArea(contour))

    # Используем Union-Find для группировки дублей
    parent = list(range(len(contours)))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Сравниваем все пары контуров
    for i in range(len(contours)):
        for j in range(i + 1, len(contours)):
            x1, y1, w1, h1 = bboxes[i]
            x2, y2, w2, h2 = bboxes[j]

            # Вычисляем пересечение bbox-ов
            xi1 = max(x1, x2)
            yi1 = max(y1, y2)
            xi2 = min(x1 + w1, x2 + w2)
            yi2 = min(y1 + h1, y2 + h2)

            if xi2 > xi1 and yi2 > yi1:
                # Есть пересечение
                intersection_area = (xi2 - xi1) * (yi2 - yi1)
                union_area = w1 * h1 + w2 * h2 - intersection_area

                # IoU (Intersection over Union)
                iou = intersection_area / union_area if union_area > 0 else 0

                # Если достаточно похожи, группируем как дубликаты
                if iou >= iou_threshold:
                    union(i, j)

    # Группируем контуры по их корневому элементу
    groups = {}
    for i in range(len(contours)):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(i)

    # Из каждой группы берём контур с максимальной площадью
    result_contours = []
    for group_indices in groups.values():
        # Найдём индекс контура с максимальной площадью
        max_idx = max(group_indices, key=lambda i: areas[i])
        result_contours.append(contours[max_idx])

    return result_contours


def _filter_convex_contours(contours, convexity_threshold=0.9):
    """
    Оставляет только выпуклые контуры.
    
    Args:
        contours: список контуров
        convexity_threshold: минимальный коэффициент выпуклости (0-1).
                           convexity = contour_area / hull_area
                           1.0 = полностью выпуклый контур
                           Рекомендуется 0.85-0.95
    
    Returns:
        Список только выпуклых контуров
    """
    filtered = []
    
    for contour in contours:
        # Проверка 1: встроенная функция OpenCV
        if cv2.isContourConvex(contour):
            filtered.append(contour)
            continue
        
        # Проверка 2: на основе соотношения площадей
        contour_area = cv2.contourArea(contour)
        if contour_area == 0:
            continue
            
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        
        if hull_area == 0:
            continue
        
        convexity = contour_area / hull_area
        
        if convexity >= convexity_threshold:
            filtered.append(contour)
    
    return filtered


def _preprocess_contour(contour, min_distance=20):
    """
    Предобработка контура: объединение всех близких вершин (не обязательно соседних).
    Может вернуть 0, 1 или несколько контуров.
    """

    if len(contour) == 0:
        return []

    # Построим ор.граф
    edges = {i: {i+1} for i in range(len(contour) - 1)}
    edges[len(contour) - 1] = {0}

    # сортировка по X для ускорения поиска близких вершин
    indexed_contour = enumerate(contour)
    sorted_indexed_contour = sorted(indexed_contour, key=lambda x: x[1][0, 0])

    # поиск близких вершин. Близкие но не соседние вершины добавляют
    # в орграф по 1 ребру в обе стороны
    for i in range(len(sorted_indexed_contour)):
        for j in range(i + 1, len(sorted_indexed_contour)):
            idx0, coords0 = sorted_indexed_contour[i]
            idx1, coords1 = sorted_indexed_contour[j]

            if (idx0 in edges[idx1]) or (idx1 in edges[idx0]):
                continue  # и так соседние

            x0, y0 = coords0[0]
            x1, y1 = coords1[0]

            if abs(x0 - x1) > min_distance:
                break

            if (y0 - y1) ** 2 + (x0 - x1) ** 2 <= min_distance ** 2:
                edges[idx0].add(idx1)
                edges[idx1].add(idx0)

    def find_shortest_path(edges, start, end):
        """
        Найти кратчайший путь по ребрам от start до end
        Возвращает массив вершин пути или None, если не найден
        """
        if start == end:
            return [start]

        queue = deque([start])
        parent = {start: None}
        distance = {start: 0}

        while queue:
            current = queue.popleft()
            if current == end:
                path = []
                node = end
                while node is not None:
                    path.append(node)
                    node = parent[node]
                path.reverse()
                return path

            for neighbor in edges[current]:
                if neighbor not in distance:
                    distance[neighbor] = distance[current] + 1
                    parent[neighbor] = current
                    queue.append(neighbor)

        return None

    def get_edge_length_squared(i, j):
        x1, y1 = contour[i, 0]
        x2, y2 = contour[j, 0]
        return (x1-x2)**2+(y1-y2)**2

    edges_sorted_by_length = []
    for v, edgs in edges.items():
        new_lengths = [(v, e, get_edge_length_squared(v, e)) for e in edgs]
        edges_sorted_by_length.extend(new_lengths)
    edges_sorted_by_length.sort(key=lambda x: x[2], reverse=True)

    result_cycles = []

    # найти все под-контуры
    for big_edge in edges_sorted_by_length:
        start = big_edge[0]
        end = big_edge[1]
        if end not in edges[start]:  # уже использовали где-то
            continue
        # find path from end to start to close the loop
        path = find_shortest_path(edges, end, start)
        if path is None:
            continue
        result_cycles.append(np.array([contour[v] for v in path]))
        # Удаляем использованные рёбра
        for j in range(1, len(path)):
            v0 = path[j - 1]
            v1 = path[j]
            edges[v0].remove(v1)
        edges[start].remove(end)

    return result_cycles
