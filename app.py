from fastapi import FastAPI, UploadFile
from core.metrics import gpu_usage, ram_usage, timed_inference
from core.analyzer import describe_image
import tempfile

app = FastAPI()

@app.post("/analyze")
async def analyze_image(file: UploadFile):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    (ram_before, ram_total) = ram_usage()
    (vram_before, vram_total) = gpu_usage()

    caption, elapsed = timed_inference(describe_image, tmp_path)

    (ram_after, _) = ram_usage()
    (vram_after, _) = gpu_usage()

    return {
        "caption": caption,
        "metrics": {
            "latency_sec": round(elapsed, 2),
            "ram_used_mb": round(ram_after - ram_before, 2),
            "vram_used_mb": round(vram_after - vram_before, 2)
        }
    }
