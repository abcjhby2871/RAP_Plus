# https://zhuanlan.zhihu.com/p/694553491
# https://zhuanlan.zhihu.com/p/698218006
from ultralytics import YOLO
from openai import OpenAI
import base64
import json
import os
import random

# 将字节串转换为字符串
def file_to_base64(file_path):
    with open(file_path, "rb") as file:
        file_content = file.read()
        base64_encoded = base64.b64encode(file_content)
        return base64_encoded.decode('utf-8')  

def get_img(file_path):
    pics_path = []
    for root, dir, files in os.walk(file_path):
        for file in files:
            if file.lower().endswith(('.jpg')):
                pics_path.append(os.path.join(root, file))
    return pics_path

def load_prompts(file_path):
    with open(file_path, 'r') as file:
        prompts = json.load(file)
    return prompts

def get_random_prompt(prompts):
    return random.choice(prompts)

def main():
    pics_path = get_img('try')
    pics_path = pics_path[4:8]
    prompts = load_prompts('prompts.json')
    
    results = []

    for pic_path in pics_path:   
        prompt = get_random_prompt(prompts)
        base_url = 'https://api.mindcraft.com.cn/v1/'
        api_key = 'MC-FF1EEDFB112E4E778F591ED51C8529C9'

        client = OpenAI(base_url=base_url, api_key=api_key)

        params = {
            "model": "Doubao-1.5-vision-pro-32k",
            "messages": [
                {
                    "role": "user",
                    "content": [# 使用 base64 编码传输
                        {
                            'type':'image',
                            'source':{
                                'data':file_to_base64(pic_path)
                            },
                        },
                        {
                            'type':'text',
                            'text':prompt,
                        },
                    ]
                }
            ],
            "temperature": 0.2,
            "max_tokens": 4000,
            "stream": True
        }

        response = client.chat.completions.create(
            model=params.get("model"),
            messages=params.get("messages"),
            temperature=params.get("temperature"),
            max_tokens=params.get("max_tokens"),
            stream=params.get("stream"),
        )

        ans = ''
        for i in response:
            content = i.choices[0].delta.content
            if not content:
                if i.usage:
                    print('\n请求花销usage:',i.usage)
                continue
            print(content,end='',flush=True)
            ans += content
        results.append({"pic_path":pic_path,"prompt":prompt,"answer":ans})

    json_path = 'data.json'
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # 如果文件不存在或文件内容不是有效的 JSON，初始化为空列表
        existing_data = []
    existing_data.extend(results)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=4)




if __name__ == '__main__':
    main()