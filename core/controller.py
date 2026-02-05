from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import numpy as np

from text_parser import set_tesseract_path
from service import DiagramParsingService


app = FastAPI(title="Diagram Parser API")
service = DiagramParsingService()


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


@app.get("/health")
async def health_check():
    """Проверка работоспособности API"""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
