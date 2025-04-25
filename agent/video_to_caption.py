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
def select_key_frame(video_path:str,max_num_frames:int, score_path:str, frame_path:str, output_file:str, ratio=1, t1=0.8, t2=-100, all_depth=2)->List[Tuple[int,Image.Image]]: #frame_id, image
    #TODO
    outs = []
    segs = []

    if not os.path.exists(output_file):
        os.makedirs(output_file)

    with open(score_path) as f:
        itm_outs = json.load(f)
    with open(frame_path) as f:
        fn_outs = json.load(f)

    for itm_out,fn_out in zip(itm_outs,fn_outs):
        nums = int(len(itm_out)/ratio)
        new_score = [itm_out[num*ratio] for num in range(nums)]
        new_fnum = [fn_out[num*ratio] for num in range(nums)]
        score = new_score
        fn = new_fnum
        num = max_num_frames
        if len(score) >= num:
            normalized_data = (score - np.min(score)) / (np.max(score) - np.min(score))
            a, b = meanstd(len(score), [dict(score=normalized_data,depth=0)], num, [fn], t1, t2, all_depth)
            segs.append(len(a))
            out = []
            if len(score) >= num:
                for s,f in zip(a,b): 
                    f_num = int(num / 2**(s['depth']))
                    topk = heapq.nlargest(f_num, range(len(s['score'])), s['score'].__getitem__)
                    f_nums = [f[t] for t in topk]
                    out.extend(f_nums)
            out.sort()
            outs.append(out)
        else:
            outs.append(fn)

    out_score_path = os.path.join(output_file, 'selected_frames.json')
    with open(out_score_path, 'w') as f:
        json.dump(outs, f)

    # sample image from the video
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frames = len(vr)

    frames_list = []
    for out in outs:
        for frame_number in out:
            if frame_number < total_frames:
                frame = vr[frame_number].asnumpy()

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb = Image.fromarray(frame_rgb)
                frames_list.append((frame_number, frame_rgb))
            else:
                print(f"error!: frame index {frame_number} out of index")
    return frames_list

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


        
