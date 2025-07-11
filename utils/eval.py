import json

def calculate_accuracies(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total_correct = 0
    group_correct = 0
    standard_total = 0
    standard_correct = 0
    defect_total = 0
    defect_correct = 0
    deep_total = 0
    deep_correct = 0

    for item in data:
        if item['ground_truth'] == item['dichotomy']:
            total_correct += 1

        if item['label'] == 'standard comparison':
            standard_total += 1
            if item['ground_truth'] == item['dichotomy']:
                standard_correct += 1
        elif item['label'] == 'defect identification':
            defect_total += 1
            if item['ground_truth'] == item['dichotomy']:
                defect_correct += 1
        elif item['label'] == 'defect diagnosis':
            deep_total += 1
            if item['ground_truth'] == item['dichotomy']:
                deep_correct += 1

    for i in range(0, len(data), 5):
        group = data[i:i+5]
        group_is_correct = True
        for item in group:
            if item['ground_truth'] != item['dichotomy']:
                group_is_correct = False
                break
        if group_is_correct:
            group_correct += 1

    single_accuracy = total_correct / len(data)
    group_accuracy = group_correct / (len(data) // 5)
    standard_accuracy = standard_correct / standard_total if standard_total > 0 else 0
    defect_accuracy = defect_correct / defect_total if defect_total > 0 else 0
    deep_accuracy = deep_correct / deep_total if deep_total > 0 else 0

    standard_group_total = 0
    standard_group_correct = 0
    defect_group_total = 0 
    defect_group_correct = 0
    deep_group_total = 0
    deep_group_correct = 0

    for i in range(0, len(data), 5):
        group = data[i:i+5]
        label = group[0]['label']
        group_is_correct = all(item['ground_truth'] == item['dichotomy'] for item in group)

        if label == 'standard comparison':
            standard_group_total += 1
            if group_is_correct:
                standard_group_correct += 1
        elif label == 'defect identification':
            defect_group_total += 1
            if group_is_correct:
                defect_group_correct += 1
        elif label == 'defect diagnosis':
            deep_group_total += 1
            if group_is_correct:
                deep_group_correct += 1

    return {
        'single_accuracy': single_accuracy,
        'group_accuracy': group_accuracy,
        'standard_comparison_accuracy': standard_accuracy,
        'defect_identification_accuracy': defect_accuracy,
        'deep_dialogue_accuracy': deep_accuracy,
        'standard_comparison_group_accuracy': standard_group_correct / standard_group_total if standard_group_total > 0 else 0,
        'defect_identification_group_accuracy': defect_group_correct / defect_group_total if defect_group_total > 0 else 0,
        'deep_dialogue_group_accuracy': deep_group_correct / deep_group_total if deep_group_total > 0 else 0
    }

def calculate_reasoning_score(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return

if __name__ == '__main__':
    accuracies = calculate_accuracies('result/baseline/qwen_1.json')
    print(f"Single accuracy: {accuracies['single_accuracy']:.4f}")
    print(f"Group accuracy: {accuracies['group_accuracy']:.4f}")
    print(f"Standard comparison accuracy: {accuracies['standard_comparison_accuracy']:.4f}")
    print(f"Defect identification accuracy: {accuracies['defect_identification_accuracy']:.4f}")
    print(f"Defect diagnosis accuracy: {accuracies['deep_dialogue_accuracy']:.4f}")
    print(f"Standard comparison group accuracy: {accuracies['standard_comparison_group_accuracy']:.4f}")
    print(f"Defect identification group accuracy: {accuracies['defect_identification_group_accuracy']:.4f}")
    print(f"Defect diagnosis group accuracy: {accuracies['deep_dialogue_group_accuracy']:.4f}")
