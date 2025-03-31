import argparse
from datasets import load_dataset
from utils.metrics import evaluate_model
from utils.inference import generate_predictions

def main(args):
    dataset = load_dataset("mbpp", split='test[:100]')
    prompts = dataset['text']
    references = dataset['code']

    predictions = generate_predictions(prompts, args.model_dir)

    scores = evaluate_model(predictions, references)
    print("Evaluation scores on first 100 MBPP examples:")
    print(scores)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="models/checkpoints/checkpoint_final")
    args = parser.parse_args()
    main(args)
