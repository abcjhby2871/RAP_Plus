# model_wrapper.py
import torch
import json
from PIL import Image
from io import BytesIO
import requests

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from transformers import TextStreamer
import sys 
sys.path.append("../..")
from detector import Detector
from retriever import ClipRetriever

class LlavaAgent:
    def __init__(self, model_path="Hoar012/RAP-LLaVA-13b", model_base=None, device="cuda"):
        disable_torch_init()
        self.model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path, model_base, self.model_name, False, False, device=device)
        self.device = device
        # setup convo mode
        if "phi" in self.model_name.lower():
            self.conv_mode = "phi3_instruct"
        elif "v1" in self.model_name.lower():
            self.conv_mode = "llava_v1"
        else:
            self.conv_mode = "llava_v0"
        self.roles = conv_templates[self.conv_mode].roles
        self.conv_templates = conv_templates
        self.detector = Detector()

    def update_database(self,database,index_path):
        with open(f"{database}/database.json", "r") as f:
            self.db_data = json.load(f)
        all_category = list({v["category"] for v in self.db_data["concept_dict"].values()})
        self.detector.model.set_classes(all_category)
        self.retriever = ClipRetriever(data_dir=database, index_path=index_path, create_index=index_path is None)

    def load_image(self, image_file: str):
        if image_file.startswith("http"):
            response = requests.get(image_file)
            return Image.open(BytesIO(response.content)).convert("RGB")
        else:
            return Image.open(image_file).convert("RGB")

    def new_conversation(self):
        conv = self.conv_templates[self.conv_mode].copy()
        return {"conv": conv, "images": [], "image_sizes": []}

    def chat(self, state: dict, user_input: str, retrieval: bool = True, topK: int = 1, image_file: str = None, temperature: float = 0.2, max_new_tokens: int = 512):
        conv = state["conv"]
        images = state["images"]
        image_sizes = state["image_sizes"]

        if image_file:
            image = self.load_image(image_file)
            image_sizes.append(image.size)
            images.append(image)

            if retrieval:
                assert hasattr(self,'retrieval'),"Please updatabase first!"
                crops = self.detector.detect_and_crop(image)
                extra_info, rag_images = self.retriever.retrieve(self.db_data, user_input, queries=crops, topK=topK)
                user_input = DEFAULT_IMAGE_TOKEN + f"\n[{extra_info}]" + user_input
                for ret_path in rag_images:
                    ret_image = self.load_image(ret_path)
                    images.append(ret_image)
                    image_sizes.append(ret_image.size)
            else:
                user_input = DEFAULT_IMAGE_TOKEN + "\n" + user_input

        conv.append_message(conv.roles[0], user_input)
        conv.append_message(conv.roles[1], None)

        image_tensor = process_images(images, self.image_processor, self.model.config)
        if isinstance(image_tensor, list):
            image_tensor = [img.to(self.device, dtype=torch.float16) for img in image_tensor]
        else:
            image_tensor = image_tensor.to(self.device, dtype=torch.float16)

        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                do_sample=temperature > 0,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                streamer=streamer,
                use_cache=True
            )
        outputs = self.tokenizer.decode(output_ids[0]).strip()
        conv.messages[-1][-1] = outputs
        return outputs, state