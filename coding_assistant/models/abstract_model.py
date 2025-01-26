from abc import ABC, abstractmethod
from typing import Any, Optional, Union

import torch

MODEL_NAME = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"


class AbstractModel(ABC):

    def __init__(self):
        if type(self) is AbstractModel:
            raise TypeError("AbstractModel cannot be instantiated directly.")

    @property
    @abstractmethod
    def base_model(self):
        pass

    def prepare_prompt_tokens(self, prompt: torch.Tensor) -> Union[list[int], torch.Tensor]:
        """
        Prepare the prompt to the input format expected from the model.

        :param prompt: A 1 x N tensor consisting of the N prompt token ids.
        :return: the prepared prompt.
        """

        return prompt

    @abstractmethod
    def generate(self, prompt_token_ids: Union[list[int], torch.Tensor], max_new_tokens: int,
                 stop_token_id: Optional[int] = None) -> Any:
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def get_name() -> str:
        raise NotImplementedError()
