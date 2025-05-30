import json

def sin_match(concept, caption):
    length = len(concept.split(' '))
    if length == 1:
        return 0
    elif length == 2:
        if concept[0] in caption or concept[1] in caption:
            return 0.5
        else:
            return 0
    elif length == 3:
        s1 = concept[0]+concept[1]
        s2 = concept[1]+concept[2]
        if s1 in caption or s2 in caption:
            return 0.667
        elif concept[0] in caption or concept[1] in caption or concept[2] in caption:
            return 0.333
        else:
            return 0

type = ["caption-llava","caption-rap-llava", "caption-ours"]
res = ["caption-2025-04-30-17:31:20", "caption-2025-04-30-18:02:49", "caption-2025-04-30-18:44:59"]

for i in range(3):
    corr = 0
    total = 0
    print(f"model:{type[i]}")
    with open(f"results/{res[i]}.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
        for item in data:
            img_path, concepts, caption = item
            concept = concepts[0].replace('_', ' ').lower()
            caption = caption.lower()
            matched = concept in caption
            gap = 0
            if matched:
                gap = 1
            else:
                gap = sin_match(concept, caption)
            corr += gap
            total += 1

    accuracy = corr / total
    print(f"Accuracy: {accuracy:.2%} ({corr:.2f}/{total})")




    """
    model:caption-llava
    Accuracy: 8.82% (30.00/340)
    model:caption-rap-llava
    Accuracy: 69.41% (236.00/340)

    model:caption-llava
    Accuracy: 51.42% (174.83/340)
    model:caption-rap-llava
    Accuracy: 83.77% (284.83/340)
    """
