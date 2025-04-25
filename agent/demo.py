#%% 在根目录下运行
import os 
home_dir = os.path.dirname(os.path.dirname(__file__))
os.chdir(home_dir)
#%% Test of retriever
from agent.video_to_caption import External_Captioner
captioner = External_Captioner()
database_root = 'mydata/database'
captioner.load_database(database_root)
#%% Test of retriever
from PIL import Image
img_path = "mydata/database/bengio.png"
img = Image.open(img_path)
captioner.retrieve(img)