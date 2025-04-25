def __f(overall_question, frame_level_captions_list, concept_list, exclude_list):
    exclude = ("However,the following concepts was not detected in the video:\n"+ str(exclude_list)) if exclude_list else ""
    return f"""We have extracted the following observations from individual frames of a video in order to answer the question:
"{overall_question}"

Throughout the video, the following external concepts were detected and may be relevant:
{concept_list}

Here are the frame-level observations (ordered by timestamp):
{frame_level_captions_list}

{exclude}

Please synthesize these into a coherent answer to the question above. You may refer to patterns or changes across time, activities, interactions, or objects involved. You are encouraged to use the external concepts if they help understand the scene better.

Return a concise but insightful answer."""

TEMPLEATE_PROMPT = {
# Frame-level Analysis
"single_frame":
{
    "system":"You are a helpful visual assistant that can understand complex visual scenes and answer detailed questions based on them. For each image frame, you will be given external information in the form of concepts and bounding boxes, as well as an overall question that needs to be answered based on this frame. Use all available information to extract useful and relevant observations.",



    "user":lambda frame_id,overall_question,concept_box_list:f"""[Frame ID: {frame_id}]
We are analyzing a video to answer the following question:
"{overall_question}"

This is one frame from the video. In this frame, we provide the following high-level concepts and their locations in the form of concepts and bounding boxes, where each bounding box is provided in normalized xyxy format (i.e., coordinates scaled between 0 and 1 relative to the image dimensions):
{concept_box_list}

Please carefully examine the image and extract any observations that might be useful for answering the question above. Be as specific as possible, and relate your observations to the provided concepts and their locations when relevant.

Format your answer as:
- Frame ID: {frame_id}
- Caption: <your detailed and relevant observation>"""
},

# Summary 
"summary":
{
    "system":"You are a skilled summarizer and reasoner. You will be given a list of detailed visual observations from individual frames of a video. Your task is to synthesize these frame-level captions to answer the original video-level question with as much insight and accuracy as possible.",


   "user": __f 
   
}
}


