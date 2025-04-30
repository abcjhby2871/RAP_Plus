import sys 
sys.path.append(".")
from agent.basic_agent import OpenAIAgent
from agent.video_to_caption import External_Captioner,select_key_frame
from agent.template import VideoAnalyzer

from PIL import Image
from typing import Optional
from dotenv import load_dotenv

class VideoAgent:
    def __init__(self):
        self.agent = OpenAIAgent()
        self.external_captioner = External_Captioner()
         
    def _frame_caption(self,img:Image.Image,cap_img:Image.Image,frame_id:int,concept_box_list:Optional[dict]=None,enhance_info:bool=False):
        query = self.analyzer.get_prompt(frame_id,concept_box_list,enhance_info)        
        response = self.agent.ask(query,[img,cap_img],use_json=True)
        self.analyzer.update(response)
        return response
    
    def _update_database(self,info:dict):
        self.external_captioner.update_database(info)
    
    def load_database(self,database_root,index_path=None):
        self.external_captioner.load_database(database_root,index_path=index_path) 

    def ask(self,prompt,video_path,enhance_info=False,**kwargs):
        self.analyzer = VideoAnalyzer(prompt,**kwargs)
        key_frame_list = select_key_frame(video_path=video_path,**kwargs)
        caption_list = []
        element_list = set()
        exclude_list = []
        for frame_id,image in key_frame_list:
            concept_box_list, caption_image = self.external_captioner.retrieve(image,prompt,frame_id=frame_id,**kwargs)
            caption_list.append(self._frame_caption(image,caption_image,frame_id,concept_box_list,enhance_info=enhance_info))
            if enhance_info is True:
                self._update_database(caption_list.pop("concept_features",dict()))
            for k in concept_box_list:
                element_list.add(k)
        for k in self.external_captioner.database:
            if k in prompt and k not in element_list:
                exclude_list.append(k)
        #! exclude_list is ignored currently.
        info_dict = dict()
        for e in element_list:
            info_dict[e] = self.external_captioner.database.get_info(e)
        query = self.analyzer.summarize(info_dict)
        response = self.agent.ask(query,None)
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
    



    
    
 