import json
from datasets import load_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5p-220m")

def filter_by_length(prompt, code):
    return len(tokenizer.encode(prompt, truncation=True)) <= 512 and len(tokenizer.encode(code, truncation=True)) <= 512

def preprocess_codexglue():
    dataset = load_dataset("code_x_glue_ct_code_to_text", "python", split='train')
    filtered_dataset = dataset.filter(lambda x: filter_by_length(x['docstring'], x['code']))

    selected_dataset = filtered_dataset.select(range(50000))
    with open('data/processed/train.jsonl', 'w', encoding='utf-8') as f:
        for ex in selected_dataset:
            json.dump({'prompt': ex['docstring'], 'code': ex['code']}, f)
            f.write('\n')

def preprocess_apps():
    dataset = load_dataset("codeparrot/apps", split='train')
    filtered_apps = []

    for ex in dataset:
        prompt = ex['question']
        solutions = ex['solutions']
        if solutions:
            code = solutions[0]
            if filter_by_length(prompt, code):
                filtered_apps.append({'prompt': prompt, 'code': code})

    with open('data/processed/train.jsonl', 'a', encoding='utf-8') as f:
        for ex in filtered_apps:
            json.dump(ex, f)
            f.write('\n')

if __name__ == "__main__":
    print("✅ Processing CodexGLUE (50k filtered examples)...")
    preprocess_codexglue()
    print("✅ CodexGLUE completed!")

    print("✅ Processing APPS (All filtered examples)...")
    preprocess_apps()
    print("✅ APPS completed! Final dataset ready in train.jsonl")
