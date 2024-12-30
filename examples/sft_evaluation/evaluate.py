"""
In the following examples, note that "model_path" needs to be the folder containing the 
HF style weights and config.json.

Example command for evaluating the dialogue summarization task using NxD.

python evaluate.py \
    --framework="nxd" \
    --nxd_inference_path="./inference" \
    --model_path="finetuned_weights" \
    --traced_model_path="./traced_model/" \
    --tokenizer_path="meta-llama/Meta-Llama-3-8B" \
    --data_dir="samsum" \
    --prompt_template=$'Summarize this dialog:\n{{dialogue}}\n---\nSummary:\n' \
    --label_template=$'{{summary}}' \
    --metric="ROUGE" \
    --sequence_length=4096 \
    --log_path="samsum_eval_results.json"

Example command for evaluating the dialogue summarization task using TnX.

python evaluate.py \
    --framework="tnx" \
    --model_path="finetuned_weights" \
    --tokenizer_path="meta-llama/Meta-Llama-3-8B" \
    --data_dir="samsum" \
    --prompt_template=$'Summarize this dialog:\n{{dialogue}}\n---\nSummary:\n' \
    --label_template=$'{{summary}}' \
    --metric="ROUGE" \
    --sequence_length=4096 \
    --log_path="samsum_eval_results.json"
"""

import json
import argparse
import torch
import pyarrow as pa

from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, enable_full_determinism

from metrics.metric_factory import MetricFactory

def load_evaluation_model(args, tokenizer):
    if args.framework == "nxd":
        from models.nxd_llama import EvaluationLlamaNxD
        return EvaluationLlamaNxD(args, tokenizer)
    elif args.framework == "tnx":
        from models.tnx_llama import EvaluationLlamaTnX
        return EvaluationLlamaTnX(args)
    else:
        raise ValueError('Invalid model framework')

def build_evaluation_dataloader(
    tokenizer,
    dataset_path,
    batch_size=1,
    prompt_template=None,
    label_template=None
):
    # Load evaluation dataset
    if dataset_path.lower().endswith("jsonl") or dataset_path.lower().endswith("json"):
        dataset = load_dataset("json", data_files=dataset_path, split="train")
    elif dataset_path.lower().endswith("arrow"):
        mmap = pa.memory_map(dataset_path)
        dataset = pa.ipc.open_stream(mmap).read_all()
    else:
        # Try loading dataset from huggingface
        dataset = load_dataset(dataset_path, split="test")
        
    if prompt_template or label_template:
        from jinja2 import Template

        def apply_templates(example):
            processed_example = {}
            if prompt_template:
                processed_example["prompt"] = Template(prompt_template).render(**example)
            if label_template:
                processed_example["label"] = Template(label_template).render(**example)
            return processed_example
        
        dataset = dataset.map(apply_templates, batched=False, num_proc = 8) 

    # Batch tokenize prompts (BOS token added automatically)
    dataset = dataset.map(
        lambda batch : tokenizer(batch["prompt"], padding=False),
        batched=True
    )

    # Hugging face collators drop all columns other than input_ids, labels, and attn_mask
    # Need custom collator to preserve label and prompt columns
    def collate_fn(records):
        batch = tokenizer.pad(records)
        batch["input_ids"] = torch.tensor(batch["input_ids"])
        batch["attention_mask"] = torch.tensor(batch["attention_mask"])
        return batch

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn
    )

def evaluate(
    model,
    tokenizer,
    dataloader,
    metric_names,
    log_path=None
):
    metrics = [MetricFactory.get_metric(name) for name in metric_names]
    evaluated_records = []

    # Batched evaluation loop
    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Evaluation"):
            gen_seqs = model.generate(batch)

            for input_ids, gen_ids, label_text, prompt in zip(batch["input_ids"], gen_seqs, batch["label"], batch["prompt"]):
                pred_ids = gen_ids[len(input_ids):] # remove prompt from output
                pred_ids = pred_ids[pred_ids != 0] # remove padding tokens on right side

                predicted_text = tokenizer.decode(
                    pred_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )

                for metric in metrics:
                    metric.update(predicted_text, label_text)

                evaluated_records.append({
                    "prompt" : prompt,
                    "prediction" : predicted_text,
                    "label" : label_text
                })

    results = {}
    for metric_name, metric in zip(metric_names, metrics):
        results[metric_name] = metric.compute()

    if log_path:
        with open(log_path, "w") as log_file:
            json.dump(evaluated_records, log_file)

    return results

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--framework",
        choices=['nxd', 'tnx'],
        default="nxd",
        help="Framework used for running the model"
    )
    parser.add_argument(
        "--skip_nxd_trace",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Traces the model at --model_path"
    )
    parser.add_argument(
        "--nxd_inference_path",
        type=str,
        help="Path to NeuronxDistributed/examples/inference"
    )
    parser.add_argument(
        "--traced_model_path",
        type=str,
        help="Path of traced NxD model",  
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Model weight and config path.",
    )
    parser.add_argument(
        "--tnx_model_precision",
        type=str,
        default="f16",
        help="Precision of TnX model",
    )
    parser.add_argument(
        "--tp_degree",
        type=int,
        default=32,
        help="Tensor parallel degree of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for deterministic behaviour",
    )
    parser.add_argument(
        "--prompt_template",
        type=str,
        help="Jinja2 template for the prompt",
    )
    parser.add_argument(
        "--label_template",
        type=str,
        help="Jinja2 template for the label",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size used for evaluation",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        help="Path for tokenizer",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Directory of dataset used for evaluation",
    )
    parser.add_argument(
        "--metric",
        type=str,
        help="Name of metric used for evaluation. Retrieved by MetricFactory class in metric.py",
    )
    parser.add_argument(
        "--log_path",
        type=str,
        help="Path to log file",
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=4096,
        help="Maximum value for total length of input and generated text",
    )
    parser.add_argument(
        "--max_input_length",
        type=int,
        default=2048,
        help="Maximum input length allowed",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature value for generation",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Top P value for generation",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=1,
        help="Default value is 1 (Greedy sampling)",
    )
    parser.add_argument(
        "--do_sample",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="enables do_sample in generate",
    )
    return parser.parse_args()

def main():
    args = parse_arguments()

    enable_full_determinism(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    tokenizer.pad_token_id = tokenizer.pad_token_type_id
    tokenizer.padding_side = 'right' # left padding currently does not work in NxD inference

    model = load_evaluation_model(args, tokenizer)

    dataloader = build_evaluation_dataloader(
        tokenizer,
        args.data_dir,
        args.batch_size,
        args.prompt_template,
        args.label_template
    )

    results = evaluate(
        model=model,
        tokenizer=tokenizer,
        dataloader=dataloader,
        metric_names=[args.metric],
        log_path=args.log_path
    )

    print(results)

if __name__ == "__main__":
    main()
