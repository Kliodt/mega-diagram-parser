from fastapi import FastAPI, File, UploadFile, Body, Query
from fastapi.responses import JSONResponse, PlainTextResponse, HTMLResponse
import cv2
import numpy as np

from text_parser import set_tesseract_path
from service import DiagramParsingService
from bpmn_graph import bpmn_graph_to_text


app = FastAPI(title="Diagram Parser API")
service = DiagramParsingService()


def _text_to_html(s: str) -> str:
    """Экранирует HTML и заменяет \\n/\\t на явные теги для корректного отображения."""
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    s = s.replace("\t", "    ")  # табуляция → 4 пробела (сохраняются в pre)
    s = s.replace("\n", "<br>")  # явный перенос строки
    return s


@app.post("/parse-diagram-image")
async def parse_diagram(file: UploadFile = File(...)):
    """
    Принимает изображение диаграммы и выполняет его парсинг.
    
    Args:
        file: Загруженное изображение диаграммы
        
    Returns:
        JSON с результатами парсинга (пока пустой)
    """
    # Читаем файл
    contents = await file.read()
    
    # Конвертируем в numpy array
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid image format"}
        )
    
    # Выполняем парсинг через сервис
    blocks, arrows, swimlanes = service.parse_diagram(image)
    
    return service.convert_to_json(blocks, arrows, swimlanes)


@app.post("/parse-bpmn-graph")
async def parse_bpmn_graph(
    body: dict = Body(...),
    format: str = Query(
        "html",
        description="Формат ответа: html — как print с отступами, text — plain text, json — JSON",
    ),
):
    """
    Принимает JSON с BPMN графом и возвращает текстовое описание алгоритма действий.

    Args:
        body: JSON с полями nodes, edges и опционально actors
        format: html — как print (сохраняет отступы и переносы), text — plain text, json — JSON

    Returns:
        Текст, HTML или JSON в зависимости от format
    """
    try:
        text = bpmn_graph_to_text(body)
        if format == "json":
            return {"text": text}
        if format == "html":
            html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>BPMN Result</title></head>
<body><pre style="white-space: pre-wrap; font-family: monospace; margin: 1em;">{_text_to_html(text)}</pre></body>
</html>"""
            return HTMLResponse(html)
        return PlainTextResponse(text, media_type="text/plain; charset=utf-8")
    except (ValueError, TypeError) as e:
        return JSONResponse(
            status_code=400,
            content={"error": str(e)}
        )


@app.get("/health")
async def health_check():
    """Проверка работоспособности API"""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    # set_tesseract_path("C:/Program Files/Tesseract-OCR/tesseract.exe")
    uvicorn.run(app, host="0.0.0.0", port=8000)
