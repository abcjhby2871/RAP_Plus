import pathlib
import json

class DataBase:
    def __init__(self,root_dir):
        self.root_dir = root_dir
        with open(f"{root_dir}/database.json") as f:
            database_ = json.load(f)
        concept_dict = dict()
        path_to_concept = dict()
        for k,v in database_["concept_dict"].items():
            assert k[0]=='<' and k[-1]=='>', f"concept {k} should be embraced by <>"
            if "image" in v:
                image = v["image"]
                if type(image) is str:
                    image = [self._get_path(image)]
                elif type(image) is list:
                    image = [self._get_path(i) for i in image]
            else:
                try:
                    image = [self._get_path(k[1:-1]+".png")]
                except:
                    image =  [self._get_path(k[1:-1]+".jpg")]
            v["image"] = image
            concept_dict[k] = v 
            for p in image:
                path_to_concept[p] = k              
        self.database = concept_dict
        self._path_to_concept = path_to_concept

    def update(self,info):
        #TODO
        raise NotImplementedError
    
    def _get_path(self,path):
        path = pathlib.Path(path) 
        if not path.is_absolute():  
            path = pathlib.Path(self.root_dir)/path
            path = path.absolute()
        return path
    
    def path_to_concept(self,path):
        return self._path_to_concept[self._get_path(path)]
    
    def __getitem__(self,concept):
        return self.database[concept]
    
    def __iter__(self):
        return iter(self.database)
    
    def get_info(self,concept):
        return {k:v for k,v in self[concept].items() if k!="image"}




        