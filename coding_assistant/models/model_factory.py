from .abstract_model import AbstractModel
from .huggingface_model import HuggingfaceModel
from .vllm_model import VllmModel
from .sglang_model import SglangModel

AVAILABLE_MODELS: set[type[AbstractModel]] = {HuggingfaceModel, VllmModel, SglangModel}


class ModelFactory:

    @staticmethod
    def get_model(name: str) -> type[AbstractModel]:
        for model in AVAILABLE_MODELS:
            if model.get_name().lower() == name.lower():
                return model
        raise NotImplementedError(f"Model {name} is not available!")

