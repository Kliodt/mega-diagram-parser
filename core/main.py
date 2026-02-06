import cv2
import numpy as np

from block_parser_any import parse_blocks_any
from block_parser_bpmn import parse_blocks
from text_parser import parse_inner_texts, set_tesseract_path
from arrow_parser import DiagramArrow, parse_arrows, visualize_connections
from swimlane_parser import process_swimlanes
from svg_converter import convert_svg_to_image, is_svg_file

if __name__ == '__main__':

    # set_tesseract_path("C:/Program Files/Tesseract-OCR/tesseract.exe")

    # image_path = "./tmp/test/diagram.svg"
    # image_path = "./tmp/test/IMG_20260205_202735_896.png"
    # image_path = "./tmp/test/Science AI - Frame 2.jpg"
    # image_path = "./tmp/test/Test 1.png"
    # image_path = "./tmp/test/Test 2.png"
    image_path = "./tmp/test/Копия_Аттракционы_Регистрация_Page_1_drawio.png"
    
    # Читаем файл и проверяем, является ли он SVG
    with open(image_path, 'rb') as f:
        file_data = f.read()
    
    if is_svg_file(file_data):
        image = convert_svg_to_image(file_data)
        if image is None:
            print("Ошибка: не удалось конвертировать SVG. Убедитесь, что установлен cairosvg")
            exit(1)
    else:
        image = cv2.imread(image_path)
    
    # Проверяем, что изображение на белом фоне
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    
    # Если фон темный (среднее значение < 127), инвертируем изображение
    if mean_brightness < 127:
        image = cv2.bitwise_not(image)

    blocks = parse_blocks(image)
    
    # Обрабатываем swimlane: группируем по высоте, определяем названия и присваиваем id блокам
    swimlanes = process_swimlanes(image, blocks, vertical_threshold=30)
    
    # Выводим информацию о swimlane
    print(f"\nНайдено swimlane: {len(swimlanes)}")
    for swimlane in swimlanes:
        print(f"  {swimlane}")
    
    # Выводим информацию о блоках с присвоенными swimlane
    print("\nБлоки и их swimlane:")
    for block in blocks:
        if block.type != 'Swimline':
            print(f"  {block.type} -> swimlane_id: {block.swimline}")

    blocks_no_swimlines = [b for b in blocks if b.type != 'Swimline']
    swimlines = [b for b in blocks if b.type == 'Swimline']



    # Визуализация bounding boxes
    # result_image = image.copy()
    # for block in blocks:
    #     x1, y1, x2, y2 = block.bbox
    #     # Рисуем прямоугольник
    #     cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #     # Добавляем текст с типом блока
    #     cv2.putText(result_image, block.type, (x1, y1 - 5), 
    #                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # cv2.imshow("Detected Blocks", result_image)
    # cv2.waitKey(-1)
    # cv2.destroyAllWindows()

    task_blocks = [b for b in blocks if b.type == 'Task']
    parse_inner_texts(image, task_blocks)

    print(task_blocks)

    arrows = parse_arrows(image, blocks_no_swimlines, proximity_threshold=30)

    # Отладочный вывод всех блоков
    for idx, block in enumerate(blocks):
        print(f"{idx}: {block.type} | bbox: {block.bbox} | swimlane: {block.swimline} | text: '{block.inner_text}'")
    
    # for arrow in arrows:
    #     print(f"Arrow: {arrow.from_box} -> {arrow.to_box}")

    image_arrows = visualize_connections(image, arrows, blocks)

    image_arrows = cv2.resize(image_arrows, (image_arrows.shape[1] // 2, image_arrows.shape[0] // 2))

    cv2.imshow("result", image_arrows)
    cv2.waitKey(-1)
    cv2.destroyAllWindows()

    # print(blocks)


    # for b in blocks:
    #     print(b)

