#%%
import os 
import os 
os.chdir('/home/test/Workspace/znchen/tianxing_project/share/NJU/NLP/RAP_Plus')
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
os.environ['HF_HOME'] = "/home/test/Workspace/znchen/tianxing_project/share/huggingface"
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
#%%
from PIL import Image,ImageDraw
import matplotlib.pyplot as plt
img = Image.open("")
h,w = img.size
color_list = ['red','yellow','blue','green','grey']
print(img.size)
def draw_image(img:Image.Image,boxes:list):
    assert len(boxes)<6
    img = img.copy()
    draw = ImageDraw.Draw(img)
    for i,box in enumerate(boxes):
        box = (int(box[0]*h),int(box[1]*w),int(box[2]*h),int(box[3]*w))
        draw.rectangle(box, fill=color_list[i])
    return img
plt.imshow(img)
#%%
import os
from PIL import Image
import matplotlib.pyplot as plt

# 文件夹路径
folder = "mydata/video3"
spacing = 10  # 每张图片之间的间隔（单位：像素）

# 加载所有 frame
frames = []
for i in range(7):
    path = os.path.join(folder, f"frame_{i}.png")
    if os.path.exists(path):
        frames.append(Image.open(path))
    else:
        print(f"Warning: {path} not found.")

# 获取最小尺寸
min_width = min(img.width for img in frames)
min_height = min(img.height for img in frames)

# 裁剪所有图片到最小尺寸
cropped_frames = [img.crop((0, 0, min_width, min_height)) for img in frames]

# 计算总尺寸（考虑间隔）
total_width = min_width * len(cropped_frames) + spacing * (len(cropped_frames) - 1)
total_height = min_height

# 新建空白画布（背景白色）
combined_img = Image.new('RGB', (total_width, total_height), color=(255, 255, 255))

# 依次粘贴每张图
x_offset = 0
for img in cropped_frames:
    combined_img.paste(img, (x_offset, 0))
    x_offset += min_width + spacing  # 每次移动宽度 + 间隔

# 显示结果
plt.figure(figsize=(20, 5))
plt.imshow(combined_img)
plt.axis('off')
plt.show()