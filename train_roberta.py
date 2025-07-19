import os
from itertools import chain

from datasets import load_dataset, Dataset
from transformers import BertConfig, BertForMaskedLM, BertTokenizerFast, DataCollatorForLanguageModeling, TrainingArguments, Trainer
from transformers.trainer_utils import EvaluationStrategy


def tokenize_dataset(dataset: Dataset, tokenizer: BertTokenizerFast) -> Dataset:
    def tokenize_texts(entries):
        tokenized_inputs = tokenizer(
            entries["text"], return_special_tokens_mask=True, truncation=False,
        )
        return tokenized_inputs

    tokenized_dataset = dataset.map(tokenize_texts, batched=True, remove_columns=["text"], num_proc=os.cpu_count())
    return tokenized_dataset


def group_dataset(dataset: Dataset, seq_len: int) -> Dataset:
    def group_texts(entries):
        concatenated_examples = {k: list(chain(*entries[k])) for k in entries.keys()}
        total_length = len(concatenated_examples[list(entries.keys())[0]])
        if total_length >= seq_len:
            total_length = (total_length // seq_len) * seq_len
        result = {
            k: [t[i: i + seq_len] for i in range(0, total_length, seq_len)]
            for k, t in concatenated_examples.items()
        }
        return result

    grouped_dataset = dataset.map(group_texts, batched=True, num_proc=os.cpu_count())
    grouped_dataset = grouped_dataset.shuffle(seed=2525)
    return grouped_dataset


def train(dataset_id: str, tokenizer_id: str, batch_size: int):
    dataset = load_dataset(dataset_id, split='train')
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_id)

    dataset = tokenize_dataset(dataset, tokenizer)
    dataset = group_dataset(dataset, tokenizer.model_max_length)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    datka = dataset.train_test_split(test_size=0.1)

    model_config = BertConfig(vocab_size=tokenizer.vocab_size, max_position_embeddings=512)
    model = BertForMaskedLM(config=model_config)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    training_args = TrainingArguments(
        output_dir="checkpoints",
        eval_strategy=EvaluationStrategy.STEPS,
        overwrite_output_dir=True,
        num_train_epochs=40,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=8,
        per_device_eval_batch_size=10,
        logging_steps=1000,
        save_steps=1000,
        load_best_model_at_end=True,
        prediction_loss_only=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=datka['train'],
        eval_dataset=datka['test'],
    )
    trainer.train()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--kind', choices=['hill', 'meadow', 'large'], type=str, required=True)
    parser.add_argument('--batch-size', type=int, required=True)
    args = parser.parse_args()

    if args.kind == 'hill':
        dataset_id = 'OneAdder/mari-bert-pretrain-hill-mari'
        tokenizer_id = 'OneAdder/hill-mari-bert-tokenizer'
    elif args.kind == 'meadow':
        dataset_id = 'OneAdder/mari-bert-pretrain-meadow-mari'
        tokenizer_id = 'OneAdder/meadow-mari-bert-tokenizer'
    elif args.kind == 'large':
        dataset_id = 'OneAdder/mari-bert-pretrain-large'
        tokenizer_id = 'OneAdder/large-mari-bert-tokenizer'

    train(dataset_id, tokenizer_id, args.batch_size)
