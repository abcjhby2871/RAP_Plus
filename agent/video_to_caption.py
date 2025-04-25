from typing import List,Tuple
from PIL import Image 
import json
import subprocess
import importlib
import sys

from detector import Detector
from retriever import ClipRetriever
from data_base import DataBase

def import_py_file(filepath, module_name='my_data_module'):
    # 创建 spec
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.data

# 假设 my_data.py 路径为 './my_data.py'


class ClipRetriever:
    def __init__(self,data_dir , index_path, create_index):
        pass 
    def retrieve_for_box(self,database, inp, detected_regions, queries ,**kwargs):
        #TODO 这里是手动做的，返回值是  
        data = import_py_file(kwargs["key_frame_config"])
        T = data[kwargs["frame_id"]]
        return T


# 对接帧选择算法，输入视频路径，返回关键帧与对应图像
def select_key_frame(video_path,**kwargs)->List[Tuple[int,Image.Image]]: #frame_id, image
    #TODO
    ret_list = []
    data = import_py_file(kwargs["key_frame_config"])
    for f in data:
        # cmd = [
        #     "ffmpeg",
        #     "-i", f"{video_path}",
        #     "-vf", f"select=eq(n\\,{f})",
        #     "-vframes", "1",
        #     f"{kwargs['outputdir']}/frame_{f}.png"
        # ]
        # subprocess.run(cmd)
        ret_list.append((f,Image.open(f"{kwargs['outputdir']}/{f}.png").convert("RGB")))
    return  ret_list

# 检索图像，查询数据库，获取外部信息
class External_Captioner:
    def __init__(self):
        self.detector = Detector()
    
    def load_database(self,database_root,index_path=None):
        self.database = DataBase(database_root)
        all_category = []
        for concept in self.database:
            cat = self.database[concept]["category"]
            if cat not in all_category:
                all_category.append(cat)
        self.detector.model.set_classes(all_category)
        if index_path is None:
            self.retriever = ClipRetriever(data_dir = database_root, index_path = index_path, create_index = True)
        else:
            self.retriever = ClipRetriever(data_dir = database_root, index_path = index_path, create_index = False)
    
    def retrieve(self,img:Image.Image,inp="",**kwargs):
        crops, detected_regions = self.detector.detect_and_crop(img)
        box_list = self.retriever.retrieve_for_box(self.database, inp, detected_regions, queries = crops, **kwargs)
        ret_list = dict()
        for k,v in box_list.items():
            ret_list[k] = (v,self.database[k]["category"])
        return box_list


        