import json
from typing import List, Dict

class VideoAnalyzer:
    def __init__(self, question: str, context_depth: int = 3, **kwargs):
        self.question = question
        self.context_depth = context_depth
        self.results: List[Dict] = []

    def get_prompt(self, frame_id: str, concept_box_desc: str, enhance_info: bool = False) -> str:
        context_str = self._get_previous_context()
        if enhance_info is True:
            info = """\n"concept_features":{"<关键概念1>":"<概念1相关特征>","<关键概念2>":"<概念2相关特征>"}"""
        else:
            info = ""
        prompt = f"""
你是一个视觉分析专家，擅长结合上下文对视频帧进行深入理解。

【帧 ID】{frame_id}
【问题】{self.question}

该帧通过颜色矩形标注了概念信息（已在另一张图像中显现）：
{concept_box_desc}

【前几帧的观察】
{context_str if context_str else "无"}

请你从图像中提取观察，并输出以下 JSON（注意用 `<概念>` 标明）：
```json
{{
  "frame_id": "{frame_id}",
  "caption": "<详细描述帧图像内容，包含标注概念>",
  "important_concepts": ["<关键概念1>", "<关键概念2>"],
  "possible_clues": ["<与问题相关的线索1>", "<可能动机或行为分析>"],{info}
  "reasoning_chain": [
    "Step 1: <观察>",
    "Step 2: <与上下文比对>",
    "Conclusion: <合理结论>"
  ]
}}
"""
        return prompt.strip()

    def _get_previous_context(self) -> str:
        ctx = self.results[-self.context_depth:]
        if not ctx:
            return ""
        summary = []
        for r in ctx:
            s = (
                f"- 帧 {r['frame_id']}:\n"
                f"  Caption: {r['caption']}\n"
                f"  Clues: {', '.join(r['possible_clues'])}\n"
                f"  Reasoning: {' -> '.join(r['reasoning_chain'])}"
            )
            summary.append(s)
        return "\n".join(summary)

    def update(self, model_result: Dict):
        self.results.append(model_result)

    def summarize(self,info) -> str:
        """生成全局总结分析（可送入 LLM）"""
        all_clues = [c for r in self.results for c in r["possible_clues"]]
        all_chains = [f"帧 {r['frame_id']}: " + " -> ".join(r["reasoning_chain"]) for r in self.results]
        return f"""
【视频总结】
全局问题：{self.question}

【外部信息】
 {info}

【所有帧线索汇总】
{json.dumps(all_clues, ensure_ascii=False, indent=2)}

【推理过程回顾】
{chr(10).join(all_chains)}

【任务】
 请你回顾之前的分析，回答全局问题，并给出优质的最终解答，最终解答不要出现分析，要与全局问题高度一致。
""".strip()