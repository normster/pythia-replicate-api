# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
from copy import deepcopy
import json
from transformers import GPTNeoXForCausalLM, AutoTokenizer
import torch


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.default_kwargs = {
            "max_new_tokens": 20,
            "batch_size": 1,
        }
        self.model = GPTNeoXForCausalLM.from_pretrained("./pythia-12b")
        self.model.half()
        if torch.cuda.is_available():
            self.model.cuda()

        self.tokenizer = AutoTokenizer.from_pretrained("./pythia-12b")
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def predict(
        self,
        inputs: str = Input(
            description="JSON-encoded input data with keys 'prompts' and 'kwargs'"
        ),
    ) -> str:
        inputs = json.loads(inputs)

        kwargs = deepcopy(self.default_kwargs)
        kwargs.update(inputs.get("kwargs", {}))

        batch_size = kwargs.pop("batch_size")
        prompts = inputs["prompts"]
        if not isinstance(prompts, list):
            prompts = [prompts]

        generations = []
        try:
            for i in range(0, len(prompts), batch_size):
                p = prompts[i:i+batch_size]
                inputs = self.tokenizer(p, return_tensors="pt", padding=True)
                kwargs["input_ids"] = inputs["input_ids"].cuda()
                kwargs["attention_mask"] = inputs["attention_mask"].cuda()
                tokens = self.model.generate(**kwargs)
                outputs = self.tokenizer.batch_decode(tokens)
                generations.extend(outputs)
        except RuntimeError as e:
            return "Error: " + str(e)

        return json.dumps(generations)
