class DiagramBlock:
    def __init__(self, type: str = None, bbox: tuple[int, int, int, int] = None):
        self.type = type
        self.bbox = bbox
        self.inner_text = ""
        self.swimline = -1
        
    def __repr__(self):
        return f'DiagramBlock: type: {self.type}; bbox: {self.bbox}; inner_text: {self.inner_text}'
