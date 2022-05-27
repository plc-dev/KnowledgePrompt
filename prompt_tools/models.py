from huggingface_hub import snapshot_download
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import os
from typing import List, Any, NamedTuple
from enum import Enum
import typing
import torch
import transformers

# more arguments for generate method:
# https://huggingface.co/docs/transformers/v4.19.2/en/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate
# https://huggingface.co/docs/transformers/internal/generation_utils
class PromptParameters(NamedTuple):
    inputs = typing.Optional[torch.Tensor]
    max_length = typing.Optional[int]
    min_length = typing.Optional[int]
    do_sample = typing.Optional[bool]
    early_stopping = typing.Optional[bool]
    num_beams = typing.Optional[int]
    temperature = typing.Optional[float]
    top_k = typing.Optional[int]
    top_p = typing.Optional[float]
    typical_p = typing.Optional[float]
    repetition_penalty = typing.Optional[float]
    bad_words_ids = typing.Optional[typing.Iterable[int]]
    force_words_ids = typing.Union[
        typing.Iterable[int],
        typing.Iterable[typing.Iterable[int]], None
    ]
    bos_token_id = typing.Optional[int]
    pad_token_id = typing.Optional[int]
    eos_token_id = typing.Optional[int]
    length_penalty = typing.Optional[float]
    no_repeat_ngram_size = typing.Optional[int]
    encoder_no_repeat_ngram_size = typing.Optional[int]
    num_return_sequences = typing.Optional[int]
    max_time = typing.Optional[float]
    max_new_tokens = typing.Optional[int]
    decoder_start_token_id = typing.Optional[int]
    use_cache = typing.Optional[bool]
    num_beam_groups = typing.Optional[int]
    diversity_penalty = typing.Optional[float]
    prefix_allowed_tokens_fn = typing.Union[
        typing.Callable[[int, torch.Tensor], typing.List[int]],
        None
    ]
    logits_processor = typing.Optional[transformers.generation_logits_process.LogitsProcessorList]
    renormalize_logits = typing.Optional[bool]
    stopping_criteria = typing.Optional[transformers.generation_stopping_criteria.StoppingCriteriaList]
    constraints = typing.Optional[typing.List[transformers.generation_beam_constraints.Constraint]]
    output_attentions = typing.Optional[bool]
    output_hidden_states = typing.Optional[bool]
    output_scores = typing.Optional[bool]
    return_dict_in_generate = typing.Optional[bool]
    forced_bos_token_id = typing.Optional[int]
    forced_eos_token_id = typing.Optional[int]
    remove_invalid_values = typing.Optional[bool]
    synced_gpus = typing.Optional[bool]
    exponential_decay_length_penalty = typing.Union[typing.Tuple[typing.Union[int, float]], None]

class OPTModelSize(str, Enum):
    M125 = "facebook/opt-125m"
    M350 = "facebook/opt-350m"
    B1 = "facebook/opt-1.3b"
    B3 = "facebook/opt-2.7b"
    B7 = "facebook/opt-6.7b"
    B13 = "facebook/opt-13b"
    B30 = "facebook/opt-30b"

class OPTUserConfig(NamedTuple):
    checkpoint = OPTModelSize

class OPTRunner:
    """
    abstraction around setting up and prompting the meta/OPT-models \n
    params:
        - user_config: OPTUserConfig \n
    returns:
        - instance of OPTRunner
    """
    __prompt_parameters = {
        "max_length": None,
        "min_length": None,
        "do_sample": None,
        "early_stopping": None,
        "num_beams": None,
        "temperature": None,
        "top_k": None,
        "top_p": None,
        "typical_p": None,
        "repetition_penalty": None,
        "bad_words_ids": None,
        "force_words_ids": None,
        "bos_token_id": None,
        "pad_token_id": None,
        "eos_token_id": None,
        "length_penalty": None,
        "no_repeat_ngram_size": None,
        "encoder_no_repeat_ngram_size": None,
        "num_return_sequences": None,
        "max_time": None,
        "max_new_tokens": None,
        "decoder_start_token_id": None,
        "use_cache": None,
        "num_beam_groups": None,
        "diversity_penalty": None,
        "prefix_allowed_tokens_fn": None,
        "logits_processor": [],
        "renormalize_logits": None,
        "stopping_criteria": [],
        "constraints": None,
        "output_attentions": None,
        "output_hidden_states": None,
        "output_scores": None,
        "return_dict_in_generate": None,
        "forced_bos_token_id": None,
        "forced_eos_token_id": None,
        "remove_invalid_values": None,
        "synced_gpus": False,
        "exponential_decay_length_penalty": None
    }

    def __init__(self, user_config: OPTUserConfig) -> None:
        self.user_config = user_config
        self.tokenizer = self.__setup_tokenizer()
        self.model = self.__setup_model()
        
    def __setup_tokenizer(self) -> Any:
        checkpoint = self.user_config["checkpoint"]
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        return tokenizer

    def __setup_model(self) -> Any:
        """
        setup model according to self.user_config
        """
        checkpoint = self.user_config["checkpoint"]
        weights_path = snapshot_download(checkpoint)
        # If the folder contains a checkpoint that isn't sharded, it needs to point to the state dict directly
        # otherwise point to the directory containing the shard
        files = os.listdir(weights_path)
        if 'pytorch_model.bin' in files:
            weights_path = os.path.join(weights_path, 'pytorch_model.bin')
        config = AutoConfig.from_pretrained(checkpoint)
        # Initializes an empty shell with the model. This is instant and does not take any RAM.
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config)
        # Initialize the model under the previous context manager breaks the tied weights.
        model.tie_weights()
        # Infer device map automatically
        device_map = infer_auto_device_map(model.model, no_split_module_classes=[
            "OPTDecoderLayer"], dtype='float16')
        if any([k == 'disk' for k in device_map.values()]):
            offload_folder = 'offload_folder'
        else:
            offload_folder = None
        if '30b' in checkpoint:
            # Set a few layers to use the disk manually to ensure enough RAM for larger checkpoints.
            device_map['decoder.layers.23'] = 'disk'
            device_map['decoder.layers.24'] = 'disk'
            device_map['decoder.layers.25'] = 'disk'
            device_map['decoder.layers.26'] = 'disk'
            device_map['decoder.layers.27'] = 'disk'
        load_checkpoint_and_dispatch(
            model.model,
            weights_path,
            device_map=device_map,
            offload_folder=offload_folder,
            dtype='float16',
            offload_state_dict=True
        )
        model.tie_weights()

        return model
    
    def prompt_model(self, prompt: str, prompt_config: PromptParameters) -> List[Any, str]:
        """
        text generation according to prompt_config and the given prompt
        """
        inputs = self.tokenizer(prompt, return_tensors="pt")
        config = {
            **self.__prompt_parameters,
            **prompt_config
        }

        output = self.model.generate(
            inputs["input_ids"].to(0),
            **config
            # repetition_penalty=0.9
            # return_dict_in_generate=True, 
            # output_scores=True
        )

        string_output = self.tokenizer.decode(output[0].tolist())

        return [output, string_output]
