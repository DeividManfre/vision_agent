from core.analyzer import Describer
from core.metrics import gpu_usage, ram_usage, timed_inference
import tempfile
from fastapi import FastAPI, UploadFile

app = FastAPI()
describer = Describer()

@app.post("/analyze")
async def analyze_image(file: UploadFile):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    result, elapsed = timed_inference(describer.describe_image, tmp_path)

    (ram_used, ram_total) = ram_usage()
    (vram_used, vram_total) = gpu_usage()

    return {
        "caption": result,
        "metrics": {
            "latency_sec": round(elapsed, 2),
            "ram_used_mb": round(ram_used, 2),
            "vram_used_mb": round(vram_used, 2),
        },
    }
