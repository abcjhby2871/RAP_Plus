# %%
import os 
os.chdir('/home/test/Workspace/znchen/tianxing_project/share/NJU/NLP/RAP_Plus')
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
os.environ['HF_HOME'] = "/home/test/Workspace/znchen/tianxing_project/share/huggingface"
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

#%%
from agent.video_to_caption import External_Captioner
database_root = '/home/test/Workspace/znchen/tianxing_project/share/NJU/NLP/RAP_Plus/mydata/database'
captioner = External_Captioner()
captioner.load_database(database_root)
# %%
from PIL import Image 
image_path = "/home/test/Workspace/znchen/tianxing_project/share/NJU/NLP/RAP_Plus/tmp/frame_175.png"
img = Image.open(image_path)
captioner.retrieve(img,"desribe the image",distance_threshold=0.6)
# %%
import logging
logging.basicConfig(
    filename="/home/test/Workspace/znchen/tianxing_project/share/NJU/NLP/RAP_Plus/tmp/llm.log",     # 日志文件路径
    level=logging.INFO,      # 日志级别（DEBUG, INFO, WARNING, ERROR, CRITICAL）
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)   
from dotenv import load_dotenv
from agent.video_agent import VideoAgent
load_dotenv("../.env")
agent = VideoAgent()
agent.load_database("mydata/database")
res = agent.ask("describe the video",video_path="mydata/video3/video.mp4",
            key_frame_config="mydata/video3/config.py",
            outputdir="mydata/video3"
            )
print(res)
# %%
