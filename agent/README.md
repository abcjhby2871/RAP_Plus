# Agent Workflow
## inference
### Step1 配置环境变量
方法使用豆包1.5-vision-pro进行测试，确保有下边的环境变量
```
OPENAI_BASEURL= https://ark.cn-beijing.volces.com/api/v3
OPENAI_API_KEY=your api
OPENAI_MODEL=doubao-1.5-vision-pro-250328
```
### Step2 预处理
TODO:如何提取关键帧与检索定位

首先要构建一个database，根目录下需要有一个database.json的文件
```
database
├── bo.png
└── database.json
```
database.json中需要有一个键叫concept_dict，格式如下，其中image需要在database的根目录下，填写路径（绝对或者相对于database的根目录），如果没有"image"键会默认使用database下名称为概念的图片(png或jpg)。
```json
{
 "concept_dict": {
    "<bo>": {
    "name": "bo",
    "image": "./bo.png", 
    "info": "<bo> is a well-groomed, medium-sized Shiba Inu with a thick, cinnamon-colored coat, cream accents, alert eyes, and a black collar.",
    "category": "dog"
    }
 }
}
```
预处理后的内容需要组织成如下的文件夹        
```
video3
├── config.py
├── frame_0.png
├── frame_200.png
├── frame_250.png
└── frame_50.png
```
```python
'''
config.py中需要有一个data字典，键为关键帧，与该目录下的frame_{id}相对应，值也是一个字典。
键为概念，值为对应的区域，采用的是xyxy形式的坐标，取左上右下角顶点，归一化到[0,1]
'''
data = { 0:{"<man>":[0.5,0.5,1.0,1.0]},
        50:{"<man>":[0.25,0.25,0.75,0.75]},
        200:{},
     250:{"<man>":[0.25,0.25,0.5,0.5]}}

```

### Step3 agent分析
```python
from agent.video_agent import VideoAgent
agent = VideoAgent()
agent.load_database("mydata/database") #加载检索库
question = "Describe the video." #你的问题
res = agent.ask(question,   
    key_frame_config="mydata/video3/config.py",    
    outputdir="mydata/video3"
            )
# outputdir是step2中预处理后的关键帧的文件夹，而key_frame_config为对应的config.py的路径
print(res)
```
