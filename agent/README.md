# Agent Workflow
agent的工作流基本搭建完成（未进行提示词的优化）
## database
我对database进行了重构，
要求database_root下有一个database.json文件
只要包含concept_dict
```json
{
    "concept_dict": {
    "<bo>": {
        "name": "bo",
        "image": "./bo.png", #绝对路径，或者相对根路径的相对路径，或者不填（但一定要保证有图像，且名称为bo.png或bo.jpg）
        "info": "<bo> is a well-groomed, medium-sized Shiba Inu with a thick, cinnamon-colored coat, cream accents, alert eyes, and a black collar.",
        "category": "dog"
    },
    "<brown-duck>": {
        "name": "brown-duck",
        "info": "<brown-duck> is a brown, plush duck toy with a green head, an yellow beak and feet, and a white stripe around its neck.",
        "category": "plush toy"
    }   
        }
}
```
用法有变
可以直接通过概念返回对应的dict，可以作为迭代器，也可以使用path_to_concept方法
```python
from data_base import DataBase
self.database = DataBase(database_root)
all_category = []
for concept in self.database:
    cat = self.database[concept]["category"]
    if cat not in all_category:
        all_category.append(cat)
```
## Retriever

```
select_key_frame(video_path=video_path,**kwargs)
# [(frame_id,PIL.Image.Image)]
[(0,img1),(50,img2)]

retrieve_for_box(self.database, inp, detected_regions, queries = crops, **kwargs)
{
    "<box>":[0.1,0.2,0.3,0.4],
    "<man>":[0.1,0.2,0.3,0.4]
}
```

确保有下边的环境变量
```
OPENAI_MODEL=
OPENAI_API_KEY=
OPENAI_BASEURL=
```
