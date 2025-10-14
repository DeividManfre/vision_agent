import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image

def describe_image(path):
    processor = AutoProcessor.from_pretrained("HuggingFaceM4/SmolVLM-Instruct")
    model = AutoModelForImageTextToText.from_pretrained("HuggingFaceM4/SmolVLM-Instruct", torch_dtype=torch.float16, device_map="auto")

    image = Image.open(path).convert("RGB")
    prompt = "Describe this image and list visible objects."

    inputs = processor(prompt, image, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=128)
    return processor.decode(output[0], skip_special_tokens=True)
