"""
Adapter for duvo-eye-1 (https://huggingface.co/duvoai/duvo-eye-1).

duvo-eye-1 is Hcompany/Holo-3.1-35B-A3B fine-tuned with a LoRA adapter on
duvoai/SynthUI for single-step GUI element grounding. The model emits
{"x": int, "y": int} in [0, 1000] relative to the input image.

This adapter queries an OpenAI-compatible endpoint. Serve the model with vLLM
before running the evaluation (bf16 weights are ~66 GB: one H200 at TP=1, or
2x80 GB at TP=2):

  vllm serve duvoai/duvo-eye-1 \
    --served-model-name duvo-eye-1 \
    --tensor-parallel-size 2 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.90 \
    --mm-processor-kwargs '{"max_pixels": 8000000}'

Then:

  python eval_screenspot_pro.py \
    --model_type duvo_eye_1 \
    --model_name_or_path duvo-eye-1 \
    --screenspot_imgs /path/to/ScreenSpot-Pro/images \
    --screenspot_test /path/to/ScreenSpot-Pro/annotations \
    --task all --inst_style instruction --language en --gt_type positive \
    --log_path results/duvo_eye_1.json

Environment variables:
  DUVO_EYE_1_ENDPOINT: OpenAI-compatible base URL (default http://127.0.0.1:8000/v1)
"""
import base64
import json
import os
import re
from io import BytesIO

import openai
from PIL import Image

DEFAULT_ENDPOINT = "http://127.0.0.1:8000/v1"

# Prompt contract from the duvo-eye-1 model card (inherited from the base
# model's element-localization recipe). Used verbatim in training and eval.
PROMPT_TEMPLATE = (
    "Localize an element on the GUI image according to the provided target "
    "and output a click position.\n"
    ' * You must output a valid JSON following the format: '
    '{"x": int 0-1000, "y": int 0-1000}\n'
    " Your target is:\n{instruction}"
)


def convert_pil_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


class DuvoEye1Model:
    def __init__(self):
        self.client = openai.OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
            base_url=os.environ.get("DUVO_EYE_1_ENDPOINT", DEFAULT_ENDPOINT),
        )
        self.model_name = "duvo-eye-1"
        self.override_generation_config = {"temperature": 0.0}

    def load_model(self, model_name_or_path="duvo-eye-1"):
        # The model runs behind an OpenAI-compatible server (see module
        # docstring); model_name_or_path is the served model name.
        self.model_name = model_name_or_path

    def set_generation_config(self, **kwargs):
        self.override_generation_config.update(kwargs)

    def _parse_point(self, response_text):
        """Parse {"x": int, "y": int} in [0, 1000] from the response."""
        try:
            pred = json.loads(response_text)
            return float(pred["x"]), float(pred["y"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass
        match = re.search(
            r'"x"\s*:\s*(\d+(?:\.\d+)?)\s*,\s*"y"\s*:\s*(\d+(?:\.\d+)?)',
            response_text,
        )
        if match:
            return float(match.group(1)), float(match.group(2))
        return None

    def ground_only_positive(self, instruction, image):
        if isinstance(image, str):
            image_path = image
            assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."
            image = Image.open(image_path).convert("RGB")
        assert isinstance(image, Image.Image), "Invalid input image."

        base64_image = convert_pil_image_to_base64(image)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                            },
                        },
                        {
                            "type": "text",
                            "text": PROMPT_TEMPLATE.format(instruction=instruction),
                        },
                    ],
                },
            ],
            temperature=self.override_generation_config["temperature"],
            max_tokens=64,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        response_text = response.choices[0].message.content

        point = self._parse_point(response_text)
        if point is not None:
            # [0, 1000] -> relative [0, 1]
            point = [point[0] / 1000.0, point[1] / 1000.0]

        return {
            "result": "positive",
            "format": "x1y1x2y2",
            "raw_response": response_text,
            "bbox": None,
            "point": point,
        }
