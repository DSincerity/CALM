from transformers import AutoConfig, PreTrainedTokenizerFast
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union


class KoGPT2TokenizerFast(PreTrainedTokenizerFast):

    def __init__(
        self, bos_token='</s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>', mask_token='<mask>', **kwargs
    ):
        super().__init__(
            bos_token='</s>',
            eos_token='</s>',
            unk_token='<unk>',
            pad_token='<pad>',
            mask_token='<mask>',
            **kwargs,
        )

    model_input_names = ["input_ids", "attention_mask"]
