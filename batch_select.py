#!/home/test/Workspace/znchen/tianxing_project/anaconda3/bin/python
import subprocess

# frames = [0,75,175,200]
# video_path = "/home/test/Workspace/znchen/tianxing_project/share/NJU/NLP/RAP_Plus/source/v1.mp4"
# output_dir = "/home/test/Workspace/znchen/tianxing_project/share/NJU/NLP/RAP_Plus/tmp"
# for f in frames:
#     cmd = [
#         "ffmpeg",
#         "-i", "/home/test/Workspace/znchen/tianxing_project/share/NJU/NLP/RAP_Plus/mydata/v1.mp4",
#         "-vf", f"select=eq(n\\,{f})",
#         "-vframes", "1",
#         f"{output_dir}/frame_{f}.png"
#     ]
#     subprocess.run(cmd)