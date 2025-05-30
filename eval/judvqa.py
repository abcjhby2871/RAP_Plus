import json


type = ['vqa-llava', 'vqa-rap-llava', 'vqa-llava-top3', 'vqa-rap-llava-top3', 
        'tqa-llava', 'tqa-rap-llava', 'tqa-llava-top3', 'tqa-rap-llava-top3']
res = ['VQA', 'VQA-re', 'VQAtop3', 'VQAtop3-re', 'TQA', 'TQA-re', 'TQAtop3', 'TQAtop3-re']

for i in range(6):
    print(f"Model: {type[i]}")
    with open(f"results/{res[i]}.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    correct = sum(1 for item in data if item["answer"] == item["correct_answer"])
    total = len(data)
    accuracy = correct / total

    print(f"Accuracy: {accuracy:.2%} ({correct}/{total})")


"""
example_database:
    Model: vqa-llava
    Accuracy: 42.35% (72/170)
    Model: vqa-rap-llava
    Accuracy: 80.59% (137/170)
    Model: vqa-ours
    Accuracy: 72.94% (124/170)
vqatop5:
    Model: vqa-llava-top5
    Accuracy: 46.47% (79/170)
    Model: vqa-rap-llava-top5
    Accuracy: 69.41% (118/170)
database_1:
    vqa:
        Model: vqa-llava
        Accuracy: 43.53% (74/170)
        Model: vqa-rap-llava
        Accuracy: 65.88% (112/170)
        Model: vqa-llava-top3
        Accuracy: 46.47% (79/170)
        Model: vqa-rap-llava-top3
        Accuracy: 71.18% (121/170)
    tqa:
        Model: tqa-llava
        Accuracy: 44.71% (76/170)
        Model: tqa-rap-llava
        Accuracy: 64.71% (110/170)
        Model: tqa-llava-top3
        Accuracy: 46.47% (79/170)
        Model: tqa-rap-llava-top3
        Accuracy: 65.29% (111/170)
"""
