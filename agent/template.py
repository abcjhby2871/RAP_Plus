# def __f(overall_question, frame_level_captions_list, concept_list, exclude_list):
#     exclude = ("However,the following concepts was not detected in the video:\n"+ str(exclude_list)) if exclude_list else ""
#     return f"""We have extracted the following observations from individual frames of a video in order to answer the question:
# "{overall_question}"

# Throughout the video, the following external concepts were detected and may be relevant:
# {concept_list}

# Here are the frame-level observations (ordered by timestamp):
# {frame_level_captions_list}

# {exclude}

# Please synthesize these into a coherent answer to the question above. You may refer to patterns or changes across time, activities, interactions, or objects involved. You are encouraged to use the external concepts if they help understand the scene better.

# Return a concise but insightful answer."""

# TEMPLEATE_PROMPT = {
# # Frame-level Analysis
# "single_frame":
# {
#     "system":"You are a helpful visual assistant that can understand complex visual scenes and answer detailed questions based on them. For each image frame, you will be given external information in the form of concepts and bounding boxes, as well as an overall question that needs to be answered based on this frame. Use all available information to extract useful and relevant observations.",



#     "user":lambda frame_id,overall_question,concept_box_list:f"""[Frame ID: {frame_id}]
# We are analyzing a video to answer the following question:
# "{overall_question}"

# This is one frame from the video. In this frame, we provide the following high-level concepts and their locations in the form of concepts and bounding boxes, where each bounding box is provided in normalized xyxy format (i.e., coordinates scaled between 0 and 1 relative to the image dimensions):
# {concept_box_list}

# Please carefully examine the image and extract any observations that might be useful for answering the question above. Be as specific as possible, and relate your observations to the provided concepts and their locations when relevant.

# Format your answer as:
# - Frame ID: {frame_id}
# - Caption: <your detailed and relevant observation>"""
# },

# # Summary 
# "summary":
# {
#     "system":"You are a skilled summarizer and reasoner. You will be given a list of detailed visual observations from individual frames of a video. Your task is to synthesize these frame-level captions to answer the original video-level question with as much insight and accuracy as possible.",
#    "user": __f 
   
# }
# }


def __f(overall_question, frame_level_captions_list, concept_list, exclude_list):
    exclude = ("然而，以下概念在视频中未被检测到：\n" + str(exclude_list)) if exclude_list else ""
    return f"""我们从视频的单帧中提取了以下观察结果，以回答以下问题：
"{overall_question}"

在整个视频中，检测到了以下外部概念，可能与问题相关：
{concept_list}

以下是按时间戳排序的帧级观察结果：
{frame_level_captions_list}

{exclude}

请将这些信息综合成一个连贯的答案来回答上述问题。您可以参考跨时间的模式或变化、活动、交互或涉及的对象。如果外部概念有助于更好地理解场景，请尽量使用它们。

返回简洁但有洞察力的答案。"""

TEMPLEATE_PROMPT = {
# 帧级分析
"single_frame":
{
    "system":"您是一个强大的视觉助手，能够理解复杂的视觉场景并基于提供的概念信息回答详细问题。对于每一帧图像，您将获得外部信息（如高级概念及其位置），以及一个需要回答的整体问题。请充分利用这些信息提取有用且相关的观察结果。",

    "user":lambda frame_id,overall_question,concept_box_list:f"""[帧 ID: {frame_id}]
我们正在分析一段视频以回答以下问题：
"{overall_question}"

这是视频中的一帧。在这一帧中，我们提供了以下高级概念及其位置信息，形式为概念和边界框，每个边界框以归一化的xyxy格式（即坐标相对于图像尺寸缩放到0到1之间）表示：
{concept_box_list}

请仔细检查图像并提取任何可能对回答上述问题有用的观察结果。描述时，请尽量使用提供的概念名称，并用尖括号 `< >` 标记这些概念。例如，如果检测到的概念是“<猫>”，请在描述中明确提到它，并结合其位置或活动进行说明。

请按照以下格式回答：
- 帧 ID: {frame_id}
- 描述: <您的详细且相关的观察结果，需包含标记的概念名称>
"""
},

# 总结
"summary":
{
    "system":"您是一位熟练的总结者和推理者。您将获得来自视频单帧的详细视觉观察结果列表。您的任务是综合这些帧级描述，尽可能准确地回答原始视频级问题。",
   "user": __f 
   
}
}