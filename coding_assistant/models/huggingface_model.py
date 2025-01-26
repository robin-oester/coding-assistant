from typing import Optional, Any, Union

import torch
from transformers import AutoModelForCausalLM

from .abstract_model import AbstractModel, MODEL_NAME


class HuggingfaceModel(AbstractModel):

    def __init__(self) -> None:
        super().__init__()
        self._base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME,
                                                                trust_remote_code=True,
                                                                attn_implementation="eager",
                                                                torch_dtype=torch.bfloat16).cuda()

    @property
    def base_model(self):
        return self._base_model

    def prepare_prompt_tokens(self, prompt: torch.Tensor) -> Union[list[int], torch.Tensor]:
        return prompt

    def generate(self, prompt_token_ids: Union[list[int], torch.Tensor], max_new_tokens: int,
                 stop_token_id: Optional[int] = None) -> Any:
        output = self._base_model.generate(prompt_token_ids, max_new_tokens=max_new_tokens, do_sample=False,
                                           num_return_sequences=1, eos_token_id=stop_token_id)

        return output[0][prompt_token_ids.size(1):]

    @staticmethod
    def get_name() -> str:
        return "huggingface"
