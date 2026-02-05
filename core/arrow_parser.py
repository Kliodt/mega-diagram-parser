import cv2
import numpy as np

from diagram_block import DiagramBlock

class DiagramArrow:
    def __init__(self, from_box_idx, to_box_idx, start_point, end_point, path):
        self.from_box = from_box_idx
        self.to_box = to_box_idx
        self.from_point = start_point
        self.to_point = end_point
        self.line_points = path


def parse_arrows(image: np.ndarray, blocks: list[DiagramBlock], proximity_threshold=30) -> list[DiagramArrow]:
    """
    Находит линии, соединяющие элементы блок-схемы.

    Args:
        image: Изображение блок-схемы (BGR или Grayscale)
        bounding_boxes: Список bounding boxes в формате (x1, y1, x2, y2)
        proximity_threshold: Максимальное расстояние от линии до края box
        debug: Показывать промежуточные результаты

    Returns:
        Список словарей с информацией о соединениях
    """
    return _find_box_connections(image, blocks, proximity_threshold)


def visualize_connections(image: np.ndarray, connections: list[DiagramArrow], blocks: list[DiagramBlock]) -> np.ndarray:
    """
    Визуализирует найденные соединения на изображении.
    """

    result = image.copy()
    
    # Рисуем соединения
    colors = [
        (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
        (0, 255, 255), (128, 0, 255), (255, 128, 0), (0, 255, 128)
    ]

    bounding_boxes = [b.bbox for b in blocks]
    for idx, box in enumerate(bounding_boxes):
        x1, y1, x2, y2 = box
        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Текст с индексом box
        label_size = cv2.getTextSize(str(idx), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(result, (x1, y1 - label_size[1] - 5), 
                        (x1 + label_size[0], y1), (0, 255, 0), -1)
        cv2.putText(result, str(idx), (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    for idx, connection in enumerate(connections):
        color = colors[idx % len(colors)]
        line_points = connection.line_points
        
        # Рисуем линию
        for i in range(len(line_points) - 1):
            cv2.line(result, line_points[i], line_points[i + 1], color, 3)
        
        # Рисуем точки начала и конца
        print(connection.from_point)
        cv2.circle(result, connection.from_point, 6, (0, 255, 0), -1)
        cv2.circle(result, connection.to_point, 6, (0, 0, 255), -1)
        
        # Добавляем текст с информацией о соединении
        if len(line_points) > 0:
            mid_idx = len(line_points) // 2
            mid_point = line_points[mid_idx]
            text = f"{connection.from_box}->{connection.to_box}"
            
            # Фон для текста
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(result, 
                         (mid_point[0] - 2, mid_point[1] - text_size[1] - 2),
                         (mid_point[0] + text_size[0] + 2, mid_point[1] + 2),
                         (255, 255, 255), -1)
            cv2.putText(result, text, mid_point, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return result


def _find_box_connections(image: np.ndarray, blocks: list[DiagramBlock], proximity_threshold):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    bounding_boxes = [b.bbox for b in blocks]

    for x1, y1, x2, y2 in bounding_boxes:
        # Заполняем прямоугольник заданным цветом
        cv2.rectangle(gray, (x1, y1), (x2, y2), (255, 255, 255), -1)

    # Применяем edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Морфологическое замыкание для соединения разрывов
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    # cv2.imshow("slkdfj", gray)
    # cv2.waitKey(-1)
    # cv2.destroyAllWindows()

    # Находим линии с помощью HoughLinesP 
    # (PI / 2 - только вертикальные и горизонтальные)
    lines = cv2.HoughLinesP(edges, 1, np.pi/2, threshold=25,
                            minLineLength=5, maxLineGap=20)

    # # Вывод изображения с найденными линиями
    # lines_image = edges.copy()
    # if lines is not None:
    #     for line in lines:
    #         x1, y1, x2, y2 = line[0]
    #         cv2.line(lines_image, (x1, y1), (x2, y2), (255, 255, 255), 1)
    
    # cv2.imshow("Found Lines", lines_image)
    # cv2.waitKey(-1)
    # cv2.destroyAllWindows()
    
    # todo: искать кривые линии

    # todo: искать пунктирные линии

    # Более удобное представление линий
    line_segments = [((line[0][0], line[0][1]), (line[0][2], line[0][3]))
                     for line in lines]
    
    # Объединяем близкие параллельные линии
    line_segments = _merge_parallel_lines(line_segments, distance_threshold=10, angle_threshold=5)

    # # Вывод изображения с line_segments
    # debug_image = image.copy()
    # for seg in line_segments:
    #     p1, p2 = seg
    #     cv2.line(debug_image, p1, p2, (0, 255, 255), 2)
    # cv2.imshow("Line Segments", debug_image)
    # cv2.waitKey(-1)
    # cv2.destroyAllWindows()

    # Строим граф линий и группируем их в пути
    paths = _build_line_paths(line_segments, tolerance=20)

    # Находим соединения между boxes
    connections = []
    for path in paths:
        start_point = path[0]
        end_point = path[-1]
        
        # Определяем, какой конец больше похож на наконечник стрелки
        start_arrow_score = _is_arrow_end(start_point, path, gray)
        end_arrow_score = _is_arrow_end(end_point, path, gray)
        
        # Стрелка указывает на тот конец, который больше похож на наконечник
        if end_arrow_score > start_arrow_score:
            # end_point - это конец стрелки (куда указывает)
            from_point = start_point
            to_point = end_point
        else:
            # start_point - это конец стрелки (куда указывает)
            from_point = end_point
            to_point = start_point
        
        from_box_idx = _find_nearest_box(from_point, bounding_boxes, proximity_threshold)
        to_box_idx = _find_nearest_box(to_point, bounding_boxes, proximity_threshold)

        # todo: path может криво определиться, поэтому добавить проверку всех
        #   точек а не только начальной и конечной

        if from_box_idx is not None and to_box_idx is not None and from_box_idx != to_box_idx:
            connections.append(DiagramArrow(from_box_idx, to_box_idx, from_point, to_point, path))
    
    return connections


def _merge_parallel_lines(line_segments: list[tuple[tuple[int, int], tuple[int, int]]], 
                         distance_threshold: float = 10, 
                         angle_threshold: float = 5) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """
    Объединяет близкие параллельные линии, которые частично перекрываются, в одну линию.
    
    Args:
        line_segments: Список линейных сегментов [(p1, p2), ...]
        distance_threshold: Максимальное расстояние между параллельными линиями для объединения
        angle_threshold: Максимальная разница в углах (градусы) для определения параллельности
    
    Returns:
        Список объединенных линейных сегментов
    """
    if not line_segments:
        return []
    
    def get_line_angle(p1, p2):
        """Вычисляет угол линии в градусах."""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return np.degrees(np.arctan2(dy, dx))
    
    def are_parallel(seg1, seg2, threshold):
        """Проверяет, являются ли две линии параллельными."""
        angle1 = get_line_angle(seg1[0], seg1[1])
        angle2 = get_line_angle(seg2[0], seg2[1])
        
        # Нормализуем углы в диапазон [0, 180)
        angle1 = angle1 % 180
        angle2 = angle2 % 180
        
        angle_diff = abs(angle1 - angle2)
        # Учитываем переход через 0/180
        angle_diff = min(angle_diff, 180 - angle_diff)
        
        return angle_diff < threshold
    
    def segments_overlap(seg1, seg2):
        """Проверяет, перекрываются ли проекции двух сегментов."""
        # Для простоты проверяем перекрытие по оси, вдоль которой линия направлена
        p1_start, p1_end = seg1
        p2_start, p2_end = seg2
        
        # Определяем ось (горизонтальная или вертикальная)
        dx1 = abs(p1_end[0] - p1_start[0])
        dy1 = abs(p1_end[1] - p1_start[1])
        
        if dx1 > dy1:  # Более горизонтальная
            # Проверяем перекрытие по X
            x1_min, x1_max = sorted([p1_start[0], p1_end[0]])
            x2_min, x2_max = sorted([p2_start[0], p2_end[0]])
            return not (x1_max < x2_min or x2_max < x1_min)
        else:  # Более вертикальная
            # Проверяем перекрытие по Y
            y1_min, y1_max = sorted([p1_start[1], p1_end[1]])
            y2_min, y2_max = sorted([p2_start[1], p2_end[1]])
            return not (y1_max < y2_min or y2_max < y1_min)
    
    def distance_between_segments(seg1, seg2):
        """Вычисляет минимальное расстояние между двумя параллельными сегментами."""
        min_dist = float('inf')
        
        # Проверяем расстояние от каждой точки одной линии до другой линии
        for point in [seg1[0], seg1[1]]:
            dist = _point_to_segment_distance(point, seg2[0], seg2[1])
            min_dist = min(min_dist, dist)
        
        for point in [seg2[0], seg2[1]]:
            dist = _point_to_segment_distance(point, seg1[0], seg1[1])
            min_dist = min(min_dist, dist)
        
        return min_dist
    
    def merge_two_segments(seg1, seg2):
        """Объединяет два сегмента в один, используя крайние точки."""
        all_points = [seg1[0], seg1[1], seg2[0], seg2[1]]
        
        # Находим две самые удаленные точки
        max_dist = 0
        result_p1, result_p2 = seg1[0], seg1[1]
        
        for i, p1 in enumerate(all_points):
            for p2 in all_points[i+1:]:
                dist = _point_distance(p1, p2)
                if dist > max_dist:
                    max_dist = dist
                    result_p1, result_p2 = p1, p2
        
        return (result_p1, result_p2)
    
    # Основной алгоритм объединения
    merged = []
    used = [False] * len(line_segments)
    
    for i, seg1 in enumerate(line_segments):
        if used[i]:
            continue
        
        current_segment = seg1
        merged_any = True
        
        while merged_any:
            merged_any = False
            
            for j, seg2 in enumerate(line_segments):
                if used[j] or i == j:
                    continue
                
                # Проверяем условия для объединения:
                # 1. Линии параллельны
                # 2. Линии близко друг к другу
                # 3. Линии перекрываются по проекции
                if (are_parallel(current_segment, seg2, angle_threshold) and
                    distance_between_segments(current_segment, seg2) <= distance_threshold and
                    segments_overlap(current_segment, seg2)):
                    
                    # Объединяем сегменты
                    current_segment = merge_two_segments(current_segment, seg2)
                    used[j] = True
                    merged_any = True
        
        merged.append(current_segment)
        used[i] = True
    
    return merged


def _point_to_segment_distance(point, seg_start, seg_end):
    """Вычисляет расстояние от точки до линейного сегмента."""
    px, py = point
    x1, y1 = seg_start
    x2, y2 = seg_end
    
    # Вектор сегмента
    dx = x2 - x1
    dy = y2 - y1
    
    if dx == 0 and dy == 0:
        # Сегмент вырожден в точку
        return _point_distance(point, seg_start)
    
    # Параметр t для проекции точки на линию
    t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
    
    # Ближайшая точка на сегменте
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy
    
    return _point_distance(point, (closest_x, closest_y))


def _point_distance(p1: tuple[int, int], p2: tuple[int, int]) -> float:
    """Вычисляет евклидово расстояние между двумя точками."""
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) ** 0.5


def _is_opposite_direction(path: list[tuple[int, int]], new_point: tuple[int, int], 
                          end_point: tuple[int, int], angle_threshold: float = 150.0) -> bool:
    """Проверяет, идет ли новый сегмент в противоположном направлении относительно текущего пути.
    
    Args:
        path: Текущий путь из точек
        new_point: Новая точка, которую хотим добавить
        end_point: Точка в конце текущего пути
        angle_threshold: Порог угла в градусах (по умолчанию 150° - почти противоположное направление)
    
    Returns:
        True, если новый сегмент идет в противоположном направлении
    """
    if len(path) < 2:
        return False
    
    # Вектор последнего сегмента пути
    prev_point = path[-2]
    vec_path = (end_point[0] - prev_point[0], end_point[1] - prev_point[1])
    
    # Вектор нового сегмента
    vec_new = (new_point[0] - end_point[0], new_point[1] - end_point[1])
    
    # Вычисляем длины векторов
    len_path = (vec_path[0]**2 + vec_path[1]**2) ** 0.5
    len_new = (vec_new[0]**2 + vec_new[1]**2) ** 0.5
    
    if len_path < 1e-6 or len_new < 1e-6:
        return False
    
    # Вычисляем косинус угла между векторами через скалярное произведение
    dot_product = vec_path[0] * vec_new[0] + vec_path[1] * vec_new[1]
    cos_angle = dot_product / (len_path * len_new)
    
    # Ограничиваем значение для избежания ошибок округления
    cos_angle = max(-1.0, min(1.0, cos_angle))
    
    # Вычисляем угол в градусах
    angle = np.degrees(np.arccos(cos_angle))
    
    # Если угол больше порога, сегменты идут в противоположных направлениях
    return angle > angle_threshold


def _build_line_paths(line_segments: list[tuple[tuple[int, int], tuple[int, int]]],
                      tolerance: int = 15) -> list[list[tuple[int, int]]]:
    """
    Строит связные пути из отдельных линейных сегментов. 
    Соединяет сегменты, концы которых находятся близко друг к другу.

    Args:
        tolerance: Максимальное расстояние между сегментами чтобы они считались единым путем

    Returns:
        список путей из 1 или нескольких сегментов
    """
    if not line_segments:
        return []

    # Нормализуем сегменты и создаем список с индексами
    segments = []
    for i, (p1, p2) in enumerate(line_segments):
        segments.append({
            'id': i,
            'p1': p1,
            'p2': p2,
            'used': False
        })

    paths = []

    # Для каждого неиспользованного сегмента строим путь
    for start_seg in segments:
        if start_seg['used']:
            continue

        # Начинаем новый путь
        path = [start_seg['p1'], start_seg['p2']]
        start_seg['used'] = True

        # Пытаемся расширить путь в обоих направлениях
        extended = True
        while extended:
            extended = False

            # Пытаемся добавить сегмент в конец пути
            end_point = path[-1]
            best_seg, best_point, best_dist = None, None, float('inf')

            for seg in segments:
                if seg['used']:
                    continue

                # Проверяем расстояние от конца пути до начала сегмента
                dist1 = _point_distance(end_point, seg['p1'])
                dist2 = _point_distance(end_point, seg['p2'])

                if dist1 < best_dist and dist1 <= tolerance:
                    # Проверяем, не идет ли сегмент в противоположном направлении
                    if not _is_opposite_direction(path, seg['p2'], end_point):
                        best_dist = dist1
                        best_seg = seg
                        # Добавляем p2, так как p1 близко к концу
                        best_point = seg['p2']

                if dist2 < best_dist and dist2 <= tolerance:
                    # Проверяем, не идет ли сегмент в противоположном направлении
                    if not _is_opposite_direction(path, seg['p1'], end_point):
                        best_dist = dist2
                        best_seg = seg
                        # Добавляем p1, так как p2 близко к концу
                        best_point = seg['p1']

            if best_seg:
                path.append(best_point)
                best_seg['used'] = True
                extended = True

            # Пытаемся добавить сегмент в начало пути
            start_point = path[0]
            best_seg, best_point, best_dist = None, None, float('inf')

            for seg in segments:
                if seg['used']:
                    continue

                dist1 = _point_distance(start_point, seg['p1'])
                dist2 = _point_distance(start_point, seg['p2'])

                if dist1 < best_dist and dist1 <= tolerance:
                    # Проверяем, не идет ли сегмент в противоположном направлении
                    # Для начала пути используем обратную логику проверки
                    if len(path) < 2 or not _is_opposite_direction([path[1], path[0]], seg['p2'], start_point):
                        best_dist = dist1
                        best_seg = seg
                        best_point = seg['p2']
                elif dist2 < best_dist and dist2 <= tolerance:
                    # Проверяем, не идет ли сегмент в противоположном направлении
                    if len(path) < 2 or not _is_opposite_direction([path[1], path[0]], seg['p1'], start_point):
                        best_dist = dist2
                        best_seg = seg
                        best_point = seg['p1']

            if best_seg:
                path.insert(0, best_point)
                best_seg['used'] = True
                extended = True

        # Упрощаем путь, удаляя промежуточные точки на прямых линиях
        simplified_path = _simplify_path(path)

        if len(simplified_path) >= 2:
            paths.append(simplified_path)

    return paths

def _simplify_path(path: list[tuple[int, int]], tolerance: float = 2.0) -> list[tuple[int, int]]:
    """
    Упрощает путь, удаляя промежуточные точки с помощью алгоритма Рамера-Дугласа-Пекера.
    """
    if len(path) <= 2:
        return path
    
    # Преобразуем в numpy array для удобства
    points = np.array(path)
    
    # Применяем упрощение
    def rdp(points, epsilon):
        if len(points) <= 2:
            return points
        
        # Находим точку с максимальным расстоянием от линии между первой и последней точкой
        start, end = points[0], points[-1]
        dists = []
        
        for i in range(1, len(points) - 1):
            dist = _point_to_line_distance(points[i], start, end)
            dists.append(dist)
        
        if not dists:
            return points
        
        max_dist = max(dists)
        max_idx = dists.index(max_dist) + 1
        
        if max_dist > epsilon:
            # Рекурсивно упрощаем обе части
            left = rdp(points[:max_idx + 1], epsilon)
            right = rdp(points[max_idx:], epsilon)
            return np.vstack([left[:-1], right])
        else:
            # Все точки близко к линии, оставляем только концы
            return np.array([start, end])
    
    simplified = rdp(points, tolerance)
    return [tuple(p) for p in simplified]


def _point_to_line_distance(point, line_start, line_end):
    """Вычисляет расстояние от точки до линии."""
    if np.all(line_start == line_end):
        return np.linalg.norm(point - line_start)
    
    # Вектор линии
    line_vec = line_end - line_start
    # Вектор от начала линии до точки
    point_vec = point - line_start
    
    # Длина линии
    line_len = np.linalg.norm(line_vec)
    line_unitvec = line_vec / line_len
    
    # Проекция точки на линию
    proj_length = np.dot(point_vec, line_unitvec)
    proj_length = np.clip(proj_length, 0, line_len)
    
    # Ближайшая точка на линии
    closest = line_start + proj_length * line_unitvec
    
    return np.linalg.norm(point - closest)

def _find_nearest_box(point: tuple[int, int], 
                      bounding_boxes: list[tuple[int, int, int, int]], 
                      threshold: int) -> int:
    """
    Находит ближайший bounding box к точке.
    """
    x, y = point
    min_distance = float('inf')
    nearest_idx = None
    
    for idx, box in enumerate(bounding_boxes):
        x1, y1, x2, y2 = box
        distance = _point_to_box_distance(x, y, x1, y1, x2, y2)
        
        if distance < min_distance and distance <= threshold:
            min_distance = distance
            nearest_idx = idx
    
    return nearest_idx

def _point_to_box_distance(px: int, py: int, 
                           x1: int, y1: int, x2: int, y2: int) -> float:
    """
    Вычисляет минимальное расстояние от точки до краев прямоугольника.
    """
    # Находим ближайшую точку на прямоугольнике
    closest_x = max(x1, min(px, x2))
    closest_y = max(y1, min(py, y2))
    
    # Вычисляем расстояние
    dx = px - closest_x
    dy = py - closest_y
    
    return np.sqrt(dx * dx + dy * dy)

def _is_arrow_end(point: tuple[int, int], path: list[tuple[int, int]], 
                  image: np.ndarray, side: int = 15) -> float:
    """Проверяет, является ли точка концом стрелки.
    
    Анализирует количество темных/светлых пикселей в квадрате вокруг точки.
    Острый конец стрелки обычно имеет больше фоновых (светлых) пикселей.
    
    Args:
        point: Точка для проверки
        path: Путь линии
        image: Изображение (grayscale)
        side: Размер стороны квадрата для анализа
    
    Returns:
        Оценка "стрелочности" (чем выше, тем больше похоже на конец стрелки)
    """
    if len(path) < 2:
        return 0.0
    
    x, y = point
    h, w = image.shape[:2]
    
    # Определяем границы квадрата вокруг точки
    half_side = side // 2
    x1 = max(0, x - half_side)
    y1 = max(0, y - half_side)
    x2 = min(w, x + half_side)
    y2 = min(h, y + half_side)
    
    # Извлекаем квадрат из изображения
    roi = image[y1:y2, x1:x2]
    
    if roi.size == 0:
        return 0.0
    
    # Подсчитываем количество светлых пикселей
    threshold = 127
    dark_pixels = np.sum(roi < threshold)
    total_pixels = roi.size
    
    # Возвращаем долю темных пикселей
    arrow_score = dark_pixels / total_pixels
    
    return arrow_score
