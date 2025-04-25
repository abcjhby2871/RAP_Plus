import os
import json
import base64
from openai import OpenAI
from typing import List, Optional, Dict, Any,Union
from PIL import Image
from io import BytesIO
import time


class OpenAIAgent:
    def __init__(
        self,
        max_retries: int = 2,
        temperature: float = 0.2,
    ):
        self.model = os.getenv("OPENAI_MODEL")
        self.max_retries = max_retries
        self.temperature = temperature
        kwargs = dict()
        if "OPENAI_BASEURL" in os.environ:
            kwargs["base_url"] = os.environ["OPENAI_BASEURL"]
        if "OPENAI_API_KEY" in os.environ:
            kwargs["api_key"] = os.environ["OPENAI_API_KEY"]
        self.client = OpenAI(**kwargs)

    def _encode_image(self, image: Image.Image) -> str:
        """Convert PIL image to base64"""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"

    def _prepare_messages(self, prompt: Union[str,dict], system_prompt: Optional[str] = None, images: Optional[List[Image.Image]] = None):
        """构造消息格式"""
        if type(prompt) is dict:
            prompt = json.dumps(prompt,ensure_ascii=False)
        content = [{"type": "text", "text": prompt}]
        message = []
        if images:
            for img in images:
                encoded = self._encode_image(img)
                content.append({"type": "image_url", "image_url": {"url": encoded}})

        if system_prompt is not None:
            message.append( {"role":"system","content":system_prompt})

        message.append({"role": "user", "content": content})

        return message

    def _validate_json(self, text: str) -> Optional[Dict[str, Any]]:
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            print(f"❌ JSON 解码失败: {e}\n{text}")
            return None

    def ask(self, prompt: Union[str,dict],  images: Optional[List[Image.Image]] = None, system_prompt: Optional[str] = None, use_json=False) -> Dict[str, Any]:
        """主函数：输入提示词和可选图像，返回结构化 JSON"""
        for attempt in range(1, self.max_retries + 1):
            print(f"🔁 尝试第 {attempt} 次...")
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=self._prepare_messages(prompt, system_prompt, images),
                    temperature=self.temperature,
                    max_tokens=1024,
                )
                reply = response.choices[0].message.content
                if use_json is True:
                    result = self._validate_json(reply)
                else:
                    result = reply
                if result is not None:
                    return result
            except Exception as e:
                print(f"⚠️ OpenAI API 请求失败: {e}")
            time.sleep(2)  # 简单退避策略
        raise RuntimeError("⛔ 所有尝试均失败，未能获取合法 JSON 响应。")
        
        