# Agent Workflow
agent的工作流基本搭建完成（未进行提示词的优化）
目前需要的是完善一下retriever，返回的值如
```
{
    "<man>":[[0.5,0.5,1.0,1.0], "person"],
    "<bo>:[[0.0,1.0.0.5,1.0],"dog"]`
},
None (or [image_list aligned with the former dict], may use in future)
```
确保有下边的环境变量
```
OPENAI_MODEL=
OPENAI_API_KEY=
OPENAI_BASEURL=
```
