import json
import os

# Input file path
input_file = '{your_path_to_mmrlhf}/MM-RLHF/dpo_pairs.jsonl'

# Output file paths
video_output_file = 'tmp/mmrlhf_v1_video.jsonl'
image_output_file = 'tmp/mmrlhf_v1_image.jsonl'

# Directories for image and video
image_dir = '{your_path_to_mmrlhf}/MM-RLHF/'
video_dir = '{your_path_to_mmrlhf}/MM-RLHF/'

# Open input file for reading
with open(input_file, 'r') as infile:
    # Open output files for writing
    with open(video_output_file, 'w') as video_file, open(image_output_file, 'w') as image_file:
        for line in infile:
            item = json.loads(line.strip())  # Read each line and parse as JSON
            
            if 'video' in item:  # If it contains a "video" field
                if video_dir not in item['video']:
                    item['video'] = os.path.join(video_dir, item['video'])
                assert os.path.exists(item['video'])
                
                # Retain only necessary elements
                item['question'] = item['prompt']
                item['response'] = item['chosen']
                item['rejected_response'] = item['rejected']
                # Delete unnecessary fields
                del item['prompt'], item['chosen'], item['rejected']
                
                # Remove all other fields except the ones we want to retain
                for key in list(item.keys()):
                    if key not in ['video', 'question', 'response', 'rejected_response']:
                        del item[key]

                # Write the filtered item to the video output file
                json.dump(item, video_file)
                video_file.write('\n')

            else:  # Otherwise save as image file
                if image_dir not in item['image']:
                    item['image'] = os.path.join(image_dir, item['image'])
                assert os.path.exists(item['image'])
                
                # Retain only necessary elements
                item['question'] = item['prompt']
                item['response'] = item['chosen']
                item['rejected_response'] = item['rejected']
                # Delete unnecessary fields
                del item['prompt'], item['chosen'], item['rejected']
                
                # Remove all other fields except the ones we want to retain
                for key in list(item.keys()):
                    if key not in ['image', 'question', 'response', 'rejected_response']:
                        del item[key]

                # Write the filtered item to the image output file
                json.dump(item, image_file)
                image_file.write('\n')
