from typing import Optional, Any, Union

import torch

import sglang as sgl

from .abstract_model import AbstractModel, MODEL_NAME


class SglangModel(AbstractModel):

    def __init__(self) -> None:
        super().__init__()

        self._base_model = None
        self.llm = sgl.Engine(model_path=MODEL_NAME, skip_tokenizer_init=True, tp_size=1)

    @property
    def base_model(self):
        return self._base_model

    def prepare_prompt_tokens(self, prompt: torch.Tensor) -> Union[list[int], torch.Tensor]:
        return prompt.squeeze().tolist()

    def generate(self, prompt_token_ids: Union[list[int], torch.Tensor], max_new_tokens: int,
                 stop_token_id: Optional[int] = None) -> Any:
        sampling_params = {"temperature": 0, "max_new_tokens": max_new_tokens, "stop_token_ids": [stop_token_id],
                           "ignore_eos": True if stop_token_id is None else False}
        output = self.llm.generate(input_ids=prompt_token_ids, sampling_params=sampling_params)

        return output["token_ids"]

    @staticmethod
    def get_name() -> str:
        return "sglang"
