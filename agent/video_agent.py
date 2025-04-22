import sys 
sys.path.append(".")
from agent.basic_agent import OpenAIAgent
from agent.template import TEMPLEATE_PROMPT
from agent.video_to_caption import External_Captioner,select_key_frame

from PIL import Image
from typing import Optional
from dotenv import load_dotenv

class VideoAgent:
    def __init__(self):
        self.agent = OpenAIAgent()
        self.external_captioner = External_Captioner()
         
    def _frame_caption(self, img:Image.Image,frame_id:int,overall_question=None,concept_box_list:Optional[dict]=None):
        query = TEMPLEATE_PROMPT["single_frame"]["user"](frame_id,overall_question,concept_box_list)
        print(query)
        response = self.agent.ask(query,[img],TEMPLEATE_PROMPT["single_frame"]["system"])
        return response
    
    def load_database(self,database_root,index_path=None):
        self.external_captioner.load_database(database_root,index_path=index_path) 

    def ask(self,prompt,video_path,**kwargs):
        key_frame_list = select_key_frame(video_path=video_path,**kwargs)
        caption_list = []
        element_list = set()
        exclude_list = []
        for frame_id,image in key_frame_list:
            concept_box_list,_ = self.external_captioner.retrieve(image,prompt,topK=2,frame_id=frame_id,**kwargs)
            caption_list.append(self._frame_caption(image,frame_id,prompt,concept_box_list))
            for k in concept_box_list:
                print(k)
                element_list.add(k)
        for k in self.external_captioner.concept_list():
            if k in prompt and k not in element_list:
                exclude_list.append(k)
        query = TEMPLEATE_PROMPT["summary"]["user"](prompt, caption_list, element_list, exclude_list)
        print(query)
        response = self.agent.ask(query,None,TEMPLEATE_PROMPT["summary"]["system"])
        return response
                
if __name__=="__main__":
    load_dotenv()
    agent = VideoAgent()
    agent.load_database("/home/test/Workspace/znchen/tianxing_project/share/NJU/NLP/RAP_Plus/mydata/database")
    res = agent.ask("describe the video",video_path="/home/test/Workspace/znchen/tianxing_project/share/NJU/NLP/RAP_Plus/mydata/v1.mp4",
              key_frame_config="/home/test/Workspace/znchen/tianxing_project/share/NJU/NLP/RAP_Plus/mydata/config.py",
              outputdir="/home/test/Workspace/znchen/tianxing_project/share/NJU/NLP/RAP_Plus/mydata/video1"
                )
    print(res)
    



    
    
 