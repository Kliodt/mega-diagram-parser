from fastapi import FastAPI, File, Form, UploadFile, Body, Query
from fastapi.responses import JSONResponse, PlainTextResponse, HTMLResponse
import cv2
import numpy as np

from text_parser import set_tesseract_path
from service import DiagramParsingService
from bpmn_graph import bpmn_graph_to_text
from svg_converter import convert_svg_to_image, is_svg_file


app = FastAPI(title="Diagram Parser API")
service = DiagramParsingService()


def _text_to_html(s: str) -> str:
    """Экранирует HTML и заменяет \\n/\\t на явные теги для корректного отображения."""
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    s = s.replace("\t", "    ")  # табуляция → 4 пробела (сохраняются в pre)
    s = s.replace("\n", "<br>")  # явный перенос строки
    return s


@app.post("/parse-diagram-image")
async def parse_diagram(file: UploadFile = File(...), output: str = Form("json")):
    """
    Принимает изображение диаграммы и выполняет его парсинг.
    
    Args:
        file: Загруженное изображение диаграммы
        output: Формат вывода ("json" или "html")
        
    Returns:
        JSON или HTML с результатами парсинга
    """
    # Читаем файл
    contents = await file.read()
    
    # Проверяем, является ли файл SVG
    if is_svg_file(contents):
        # Конвертируем SVG в растровое изображение
        image = convert_svg_to_image(contents)
        if image is None:
            return JSONResponse(
                status_code=400,
                content={"error": "Failed to convert SVG image. Make sure cairosvg is installed."}
            )
    else:
        # Конвертируем обычное изображение в numpy array
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid image format"}
            )
    
    # Выполняем парсинг через сервис
    blocks, arrows, swimlanes = service.parse_diagram(image)
    
    json_result = service.convert_to_json(blocks, arrows, swimlanes)
    
    if output == "html":
        # Конвертируем JSON в читаемый текст
        text = bpmn_graph_to_text(json_result)
        html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>Diagram Parser Result</title></head>
<body><pre style="white-space: pre-wrap; font-family: monospace; margin: 1em;">{_text_to_html(text)}</pre></body>
</html>"""
        return HTMLResponse(html)
    
    return json_result


@app.get("/health")
async def health_check():
    """Проверка работоспособности API"""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    # set_tesseract_path("C:/Program Files/Tesseract-OCR/tesseract.exe")
    uvicorn.run(app, host="0.0.0.0", port=8000)
