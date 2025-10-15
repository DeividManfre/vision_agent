from core.analyzer import Describer
from core.metrics import SystemMetrics
from fastapi import FastAPI, UploadFile
import tempfile

app = FastAPI()
describer = Describer()
metrics = SystemMetrics()

@app.post("/analyze")
async def analyze_image(file: UploadFile):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
        
        result, elapsed, ram_used, vram_used = metrics.timed_inference(
            describer.describe_image, tmp_path
        )
        
        return {
            "caption": result,
            "metrics": {
                "latency_sec": elapsed,
                "ram_used_mb": ram_used,
                "vram_used_mb": vram_used
            }
        }
        