# Vision Agent

**Vision Agent** is a lightweight AI system that analyzes and describes images while tracking real-time system efficiency.

### What it does
- **Generates detailed image captions** using SmolVLM or Qwen-VL models  
- **Identifies visual elements** and contextual objects  
- **Measures inference metrics** (latency, RAM, and VRAM usage)  
- **Exports results** to Markdown or an interactive Streamlit dashboard  

Built to showcase **AI integration, GPU efficiency, and DevOps monitoring** — ideal for portfolios, benchmarks, and production-ready demos.

---

## Tech Stack
- **Python** · FastAPI / Streamlit  
- **Models:** SmolVLM · Qwen2.5-VL  
- **Metrics:** psutil · pynvml  
- **Utilities:** OpenCV · Pillow · Torch  

---

## Example Usage


```
 uvicorn app:app --reload --port 6969
```
[link](http://127.0.0.1:6969/docs)

<video width="640" controls>
  <source src="video/demo.gif" type="video/mp4">
</video>