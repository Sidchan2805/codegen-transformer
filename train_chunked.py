import argparse, os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset

def preprocess(examples, tokenizer):
    inputs = tokenizer(examples['prompt'], max_length=512, truncation=True, padding='max_length')
    labels = tokenizer(examples['code'], max_length=512, truncation=True, padding='max_length')
    inputs["labels"] = labels["input_ids"]
    return inputs

def train_on_chunk(chunk_file, model, tokenizer, chunk_idx, args):
    dataset = load_dataset("json", data_files=chunk_file, split='train')
    dataset = dataset.map(lambda x: preprocess(x, tokenizer), batched=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=f"{args.checkpoints_dir}/checkpoint_{chunk_idx}",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=args.epochs,
        learning_rate=5e-5,
        logging_steps=500,
        save_steps=5000,
        fp16=True
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer
    )

    trainer.train()
    trainer.save_model(f"{args.checkpoints_dir}/checkpoint_{chunk_idx}")

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    chunks = sorted(os.listdir(args.chunks_dir))
    os.makedirs(args.checkpoints_dir, exist_ok=True)

    for idx, chunk in enumerate(chunks, 1):
        chunk_file = os.path.join(args.chunks_dir, chunk)
        print(f"✅ Training on {chunk}...")
        train_on_chunk(chunk_file, model, tokenizer, idx, args)
        # Load model from latest checkpoint
        model = AutoModelForSeq2SeqLM.from_pretrained(f"{args.checkpoints_dir}/checkpoint_{idx}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Salesforce/codet5p-220m")
    parser.add_argument("--chunks_dir", default="data/processed/chunks")
    parser.add_argument("--checkpoints_dir", default="models/checkpoints")
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    main(args)
