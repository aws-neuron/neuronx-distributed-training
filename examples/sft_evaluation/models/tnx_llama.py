from models.eval_model import EvaluationModel

class EvaluationLlamaTnX(EvaluationModel):
    def __init__(self, args):
        from transformers_neuronx.llama.model import LlamaForSampling # type: ignore

        self.generation_args = {
            "temperature" : args.temperature,
            "sequence_length" : args.sequence_length,
            "top_p" : args.top_p,
            "do_sample" : args.do_sample,
            "top_k": args.top_k
        }

        self.model = LlamaForSampling.from_pretrained(
            args.model_path,
            batch_size=args.batch_size,
            amp=args.tnx_model_precision,
            tp_degree=args.tp_degree
        )
        
        self.model.to_neuron()
        self.model.eval()

    def generate(self, batch):
        return self.model.sample(
            input_ids=batch['input_ids'],
            **self.generation_args
        )