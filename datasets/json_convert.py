import random
import json
import re
from json_loader import extract_defect_data_from_file

def classify_data(item_list):
    defect_identification_list = []
    defect_diagnosis_list = []
    standard_comparison_list = []

    for entry in item_list:
        if entry['pair_type'] == 'defect identification':
            defect_identification_list.append(entry)
        elif entry['pair_type'] == 'defect diagnosis':
            defect_diagnosis_list.append(entry)
        elif entry['pair_type'] == 'standard comparison':
            standard_comparison_list.append(entry)
    
    return defect_identification_list, defect_diagnosis_list, standard_comparison_list

def split_data(data, train_ratio=0.2):
    random.shuffle(data)
    split_index = int(len(data) * train_ratio)
    train_set = data[:split_index]
    test_set = data[split_index:]
    return train_set, test_set

def merge_data(item_list, train_ratio=0.2):
    defect_identification_list, defect_diagnosis_list, standard_comparison_list = classify_data(item_list)
    train_set = []
    test_set = []
    for data_list in [defect_identification_list, defect_diagnosis_list, standard_comparison_list]:
        train_class, test_class = split_data(data_list, train_ratio)
        train_set.extend(train_class)
        test_set.extend(test_class)
    return train_set, test_set

def modify_json(input_data, output_file_path):
    modified_data = []
    for entry in input_data:
        test_image_value = "dataset/MANTA_SETTING2_DATASET/ImageData/" + entry.get('domain', '') + '/' + entry.get('test_image', '')
        image_value = [test_image_value]
        messages = []
        pair_type = entry.get('pair_type', '')
        conclusion_text = entry.get('conclusion_en', '')

        for idx, qa in enumerate(entry.get('question_answer', [])):
            question = qa.get('question_en', '')
            answer = qa.get('answer', None)

            if pair_type == "defect diagnosis":
                conclusion = qa.get('conclusion_en', '')
                gpt_value = f"{'True' if answer else 'False'}. {conclusion}"
            elif pair_type in ['defect identification', 'standard comparison']:
                last_word = question.split()[-1].strip("?")
                pattern = rf"[^.?!]*\b{last_word}\b[^.?!]*\."
                match = re.search(pattern, conclusion_text, re.IGNORECASE | re.DOTALL)
                if match:
                    matched_sentence = match.group(0).strip()
                    gpt_value = f"{'True' if answer else 'False'}. {matched_sentence}"
                else:
                    gpt_value = f"{'True' if answer else 'False'}. {conclusion_text.strip()}"
            else:
                continue

            messages.append({
                "content": f"{'<image>' if idx == 0 else ''}{question}",
                "role": "user"
            })

            messages.append({
                "content": gpt_value,
                "role": "assistant"
            })

        modified_entry = {
            "messages": messages,
            "images": image_value
        }

        modified_data.append(modified_entry)

    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(modified_data, f, ensure_ascii=False, indent=4)
    return json.dumps(modified_data, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    file_path1 = '../../MANTA_SETTING2_DATASET/JsonData/electronics_QA.json'
    file_path2 = '../../MANTA_SETTING2_DATASET/JsonData/agriculture_QA.json'
    file_path3 = '../../MANTA_SETTING2_DATASET/JsonData/groceries_QA.json'
    file_path4 = '../../MANTA_SETTING2_DATASET/JsonData/mechanics_QA.json'
    file_path5 = '../../MANTA_SETTING2_DATASET/JsonData/medicine_QA.json'
    item_list = []
    item_list.extend(extract_defect_data_from_file(file_path1))
    item_list.extend(extract_defect_data_from_file(file_path2))
    item_list.extend(extract_defect_data_from_file(file_path3))
    item_list.extend(extract_defect_data_from_file(file_path4))
    item_list.extend(extract_defect_data_from_file(file_path5))
    train_set, test_set = merge_data(item_list)
    print(len(train_set), len(test_set))
    modify_json(train_set, '../latest_data/train.json')
    modify_json(test_set, '../latest_data/test.json')
