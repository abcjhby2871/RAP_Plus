import requests
import argparse

import os
import base64
import subprocess
import openai
from glob import glob
from PIL import Image


class VideoQAAgent:
    def __init__(self, api_key: str, model: str = "gpt-4-vision-preview"):
        self.model = model
        openai.api_key = api_key
        self.captions = {}  # 存 frame: caption

    def extract_frames(self, video_path: str, fps: float = 0.5, out_dir: str = "frames"):
        os.makedirs(out_dir, exist_ok=True)
        subprocess.run([
            "ffmpeg", "-i", video_path, "-vf", f"fps={fps}",
            f"{out_dir}/frame_%04d.jpg", "-hide_banner", "-loglevel", "error"
        ])
        self.frames = sorted(glob(f"{out_dir}/frame_*.jpg"))
        print(f"[INFO] Extracted {len(self.frames)} frames.")

    def encode_image(self, image_path: str):
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def caption_frame(self, image_path: str, prompt: str = "Describe this image."):
        session_id = start_session(args.base_url)
        caption = chat(session_id, user_input, image_path=args.image, base_url=args.base_url)
        return caption

    def build_frame_caption_index(self):
        print("[INFO] Generating captions for all frames...")
        for path in self.frames:
            cap = self.caption_frame(path)
            self.captions[path] = cap
            print(f"[{os.path.basename(path)}] {cap}")

    def answer_question(self, question: str):
        # 把所有 caption 拼接作为上下文
        context = "\n".join(
            f"{i+1}. {cap}" for i, cap in enumerate(self.captions.values())
        )
        full_prompt = f"""This is a list of video frame captions:

{context}

Now answer the following question based on the frames above:
"{question}"
"""
        response = openai.ChatCompletion.create(
            model="gpt-4",  # 用文本模型回答问题
            messages=[
                {"role": "system", "content": "You are a helpful assistant that analyzes video content."},
                {"role": "user", "content": full_prompt}
            ],
            max_tokens=512
        )
        return response["choices"][0]["message"]["content"]
    
def start_session(base_url="http://localhost:8000"):
    response = requests.post(f"{base_url}/start")
    if response.ok:
        session_id = response.json()["session_id"]
        print(f"✅ New session started: {session_id}")
        return session_id
    else:
        raise RuntimeError("Failed to start session")

def update_database(db_path, index_path=None,base_url="http://localhost:8000"):
    payload = {
        "database": db_path,
        "index_path": index_path
    }
    resp = requests.post(f"{base_url}/update_db", json=payload)
    print(resp.json())

def chat(session_id, user_input, image_path=None,topK=1, base_url="http://localhost:8000"):
    files = {
        "session_id": (None, session_id),
        "user_input": (None, user_input),
        "topK": (None,topK)
    }

    if image_path:
        files["image"] = (image_path, open(image_path, "rb"), "image/jpeg")

    response = requests.post(f"{base_url}/chat", files=files)
    if response.ok:
        answer = response.json()["answer"]
        return answer
        # print(f"\n🧠 Assistant: {answer}")
    else:
        print("❌ Error:", response.text)
        raise Exception

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", type=str, default="http://localhost:8000", help="Base URL of the server")
    parser.add_argument("--image", type=str, default=None, help="Path to image file")
    args = parser.parse_args()

    session_id = start_session(args.base_url)

    print("\n🔁 Start chatting! Press Ctrl+C to exit.")
    try:
        while True:
            user_input = input("\n🧑 You: ")
            if not user_input.strip():
                continue
            chat(session_id, user_input, image_path=args.image, base_url=args.base_url)
    except KeyboardInterrupt:
        print("\n👋 Bye!")