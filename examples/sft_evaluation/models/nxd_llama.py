from models.eval_model import EvaluationModel

class EvaluationLlamaNxD(EvaluationModel):
    def __init__(self, args, tokenizer):
        import os
        import sys
        
        nxd_inference_path = os.path.realpath(args.nxd_inference_path)
        sys.path.insert(0, nxd_inference_path)

        from llama2.llama2_runner import LlamaRunner # type: ignore
        from transformers import GenerationConfig

        self.generation_config = GenerationConfig(
            bos_token_id = tokenizer.bos_token_id,
            eos_token_id = tokenizer.eos_token_id,
            pad_token_id = tokenizer.pad_token_id,
            do_sample = args.do_sample,
            temperature = args.temperature,
            max_length = args.sequence_length,
            top_p = args.top_p,
            top_k = args.top_k
        )

        self.runner = LlamaRunner(
            model_path = args.model_path,
            tokenizer_path = args.tokenizer_path,
            generation_config= self.generation_config
        )

        if not args.skip_nxd_trace:
            self.runner.trace(
                traced_model_path=args.traced_model_path,
                tp_degree=args.tp_degree,
                batch_size=args.batch_size,
                max_prompt_length=args.max_input_length,
                sequence_length=args.sequence_length,
                on_device_sampling = True
            )

        self.model = self.runner.load_neuron_model(args.traced_model_path)
        self.model.eval()

    def generate(self, batch):
        gen_seqs = self.model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            generation_config=self.generation_config,
            max_length=self.model.config.max_length,
        )
        self.model.reset()
        return gen_seqs