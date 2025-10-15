import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image


class Describer:
    def __init__(self, model_name=None):
        self.model_name = model_name or "Qwen/Qwen2.5-VL-3B-Instruct"
        print(f"[INFO] Loading model: {self.model_name}")

        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        print("[INFO] Model loaded successfully and ready for inference.")

    def describe_image(self, image_path):
        image = Image.open(image_path).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Describe this image and list visible objects."}
                ]
            }
        ]

        text_prompt = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text_prompt],
            images=[image],
            return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=128)

        caption = self.processor.batch_decode(output, skip_special_tokens=True)[0]

        print(f"[DONE] Caption generated: {caption}")
        return caption
