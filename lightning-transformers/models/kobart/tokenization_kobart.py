from transformers import PreTrainedTokenizerFast
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union
from tokenizers.processors import TemplateProcessing


class KoBartTokenizerFast(PreTrainedTokenizerFast):

    def __init__(
        self, bos_token='<s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>', mask_token='<mask>', **kwargs
    ):
        super().__init__(
            bos_token='<s>',
            eos_token='</s>',
            unk_token='<unk>',
            pad_token='<pad>',
            mask_token='<mask>',
            **kwargs,
        )

        self._tokenizer.post_processor = TemplateProcessing(
            single="<s> $A </s>",
            pair="<s> $A </s> <s> $B </s>",
            special_tokens=[("<s>", self._tokenizer.token_to_id('<s>')), ("</s>", self._tokenizer.token_to_id('</s>'))],
        )

    model_input_names = ["input_ids", "attention_mask"]
