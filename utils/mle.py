import openai
import json
from pathlib import Path
import time

openai.api_key = "your-api-key"

EXAMPLES = [
    {
        "question": "Does this object retain its expected load-bearing capacity?",
        "reference_answer": "No. The central hole weakens the object by removing material and creating a stress concentration point.",
        "test_answer": "No, rusting is apparent on the metal plate, indicating corrosion that compromises its structural integrity.",
        "scores": {"accuracy": 5, "localization": 5, "logic": 5}
    },
    {
        "question": "Can this component be used normally?",
        "reference_answer": "Yes, there are no issues observed with this object, consequently it functions as expected.",
        "test_answer": "Yes, there are just some spots on the bracket where the coating was missed, but this doesn't affect its normal use.",
        "scores": {"accuracy": 5, "localization": 5, "logic": 5}
    },
    {
        "question": "Is it likely to lead to spoilage of medicines?",
        "reference_answer": "No, the images appear uniform with only minor variations, likely due to lighting.",
        "test_answer": "No, the scratch anomaly is present, but it is unlikely to cause medicine spoilage.",
        "scores": {"accuracy": 5, "localization": 5, "logic": 5}
    },
    {
        "question": "Can this component be assembled correctly?",
        "reference_answer": "Yes, the rough edges visible on this part do not interfere with assembly.",
        "test_answer": "No, the gate mark deformation seen from multiple angles will severely impact this object's assembly.",
        "scores": {"accuracy": 1, "localization": 5, "logic": 5}
    }
]

def build_prompt(examples, eval_input):
    prompt = (
        "You are an expert evaluator using MLE (Multidimensional Logic Evaluation).\n"
        "You must score answers from 1 to 5 on three aspects:\n"
        "Accuracy (final conclusion alignment), Localization (visual evidence identification), Logic (reasoning clarity).\n"
        "Here are examples:\n"
    )
    for ex in examples:
        prompt += (
            f"\nQ: {ex['question']}\n"
            f"Reference Answer: {ex['reference_answer']}\n"
            f"Test Answer: {ex['test_answer']}\n"
            f"Scores:\n"
            f"- Accuracy: {ex['scores']['accuracy']}\n"
            f"- Localization: {ex['scores']['localization']}\n"
            f"- Logic: {ex['scores']['logic']}\n"
            f"- Average: {sum(ex['scores'].values()) / 3:.2f}\n"
        )
    prompt += (
        f"\nEvaluate the following:\n"
        f"Q: {eval_input['question']}\n"
        f"Reference Answer: {eval_input['reference_answer']}\n"
        f"Test Answer: {eval_input['test_answer']}\n"
        f"Scores:"
    )
    return prompt

def parse_response(text):
    scores = {"accuracy": 0, "localization": 0, "logic": 0, "average": 0}
    for line in text.splitlines():
        parts = line.lower().strip().split(":")
        if len(parts) == 2:
            key, val = parts[0].strip(), parts[1].strip()
            try:
                if "accuracy" in key:
                    scores["accuracy"] = float(val)
                elif "localization" in key:
                    scores["localization"] = float(val)
                elif "logic" in key:
                    scores["logic"] = float(val)
                elif "average" in key:
                    scores["average"] = float(val)
            except:
                continue
    return scores

def mle(input_item):
    prompt = build_prompt(EXAMPLES, input_item)
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response['choices'][0]['message']['content']

def evaluate_batch_from_json(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        items = json.load(f)

    results = []
    for i, item in enumerate(items):
        print(f"Evaluating item {i+1}/{len(items)}...")
        try:
            raw_text = mle(item)
            scores = parse_response(raw_text)
        except Exception as e:
            print(f"Error on item {i}: {e}")
            scores = {"accuracy": 0, "localization": 0, "logic": 0, "average": 0}
        results.append({**item, **scores})
        time.sleep(1)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    input_path = "input.json"
    output_path = "mle_evaluation_result.json"
    evaluate_batch_from_json(input_path, output_path)
    print("Evaluation complete. Results saved to", output_path)