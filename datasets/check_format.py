import json
import os
from PIL import Image

def check_json_format(json_data, log_file="log.txt"):
    errors = []
    for item in json_data:
        test_img = item.get("test_image", "unknown")
        if "pair_type" not in item:
            errors.append(f"{test_img}: Missing 'pair_type' field")
        else:
            valid_pair_types = ["standard comparison", "defect identification", "defect diagnosis"]
            if item["pair_type"] not in valid_pair_types:
                errors.append(f"{test_img}: Invalid 'pair_type' value: {item['pair_type']}")
        for field in ["test_image", "reference_image"]:
            if field not in item:
                errors.append(f"{test_img}: Missing '{field}' field")
                continue
            try:
                img = Image.open(item[field])
                img.close()
            except Exception as e:
                errors.append(f"{test_img}: Cannot open image file {item[field]}: {str(e)}")
        if "question_answer" not in item:
            errors.append(f"{test_img}: Missing 'question_answer' field")
        else:
            qa_list = item["question_answer"]
            if len(qa_list) != 5:
                errors.append(f"{test_img}: 'question_answer' should have exactly 5 items, found {len(qa_list)}")
            for qa in qa_list:
                for field in ["question_cn", "question_en", "answer_options"]:
                    if field not in qa:
                        errors.append(f"{test_img}: Missing '{field}' field in a question-answer pair")
                if "answer_options" in qa:
                    options = qa["answer_options"]
                    if len(options) != 4:
                        errors.append(f"{test_img}: Each 'answer_options' should have exactly 4 items, found {len(options)}")
                    for opt in options:
                        for field in ["answer_cn", "answer_en"]:
                            if field not in opt:
                                errors.append(f"{test_img}: Missing '{field}' field in an answer option")
    if errors:
        with open(log_file, "a", encoding="utf-8") as f:
            for error in errors:
                f.write(error + "\n")
        return False
    return True

if __name__ == '__main__':
    categories = ["bracket_black", "bracket_brown", "bracket_white", "connector", "metal_plate", "tubes"]
    categorie = categories[5]
    json_path = f"MPDD/{categorie}/{categorie}_fixed.json"
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    is_valid = check_json_format(data)
    if is_valid:
        print("JSON format check passed.")
    else:
        print("Format issues found. Please check the log.txt file.")
