from typing import Optional, Any, Union

import torch
from vllm import LLM, SamplingParams

from .abstract_model import AbstractModel, MODEL_NAME


class VllmModel(AbstractModel):

    def __init__(self) -> None:
        super().__init__()

        self._base_model = None

        # set 'enforce_eager=False' to enable cuda graph capturing
        self.llm = LLM(model=MODEL_NAME, tensor_parallel_size=1, trust_remote_code=True,
                       device="cuda", skip_tokenizer_init=True,
                       max_model_len=8192, enforce_eager=True)

    @property
    def base_model(self):
        return self._base_model

    def prepare_prompt_tokens(self, prompt: torch.Tensor) -> Union[list[int], torch.Tensor]:
        return prompt.squeeze().tolist()

    def generate(self, prompt_token_ids: Union[list[int], torch.Tensor], max_new_tokens: int,
                 stop_token_id: Optional[int] = None) -> Any:

        sampling_params = SamplingParams(temperature=0, max_tokens=max_new_tokens, stop_token_ids=[stop_token_id],
                                         ignore_eos=True if stop_token_id is None else False)

        output = self.llm.generate(prompt_token_ids=[prompt_token_ids], sampling_params=sampling_params)

        return output[0].outputs[0].token_ids

    @staticmethod
    def get_name() -> str:
        return "vllm"
