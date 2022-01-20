# -*- coding: utf-8 -*-
from .rte_dataset import RTEAutoInferenceDataset, RTEAutoInferenceReverseDataset, RTEAutoInferenceSignalDataset, \
    RTET5InferenceDataset, RTET5InferenceReverseDataset, RTET5InferenceSignalDataset
from .mnli_dataset import MNLIAutoInferenceDataset, MNLIAutoInferenceReverseDataset, MNLIAutoInferenceSignalDataset, \
    MNLIT5InferenceDataset, MNLIT5InferenceReverseDataset, MNLIT5InferenceSignalDataset
from .qnli_dataset import QNLIAutoInferenceDataset, QNLIAutoInferenceReverseDataset, QNLIAutoInferenceSignalDataset, \
    QNLIT5InferenceDataset, QNLIT5InferenceReverseDataset, QNLIT5InferenceSignalDataset
from .qqp_dataset import QQPAutoInferenceDataset, QQPAutoInferenceReverseDataset, QQPAutoInferenceSignalDataset, \
    QQPT5InferenceDataset, QQPT5InferenceReverseDataset, QQPT5InferenceSignalDataset
from .mrpc_dataset import MRPCAutoInferenceDataset, MRPCAutoInferenceReverseDataset, MRPCAutoInferenceSignalDataset, \
    MRPCT5InferenceDataset, MRPCT5InferenceReverseDataset, MRPCT5InferenceSignalDataset
from .klue_nli_dataset import KlueNLIAutoInferenceDataset, KlueNLIAutoInferenceReverseDataset, \
    KlueNLIAutoInferenceSignalDataset
from .klue_sts_dataset import KlueSTSAutoInferenceDataset, KlueSTSAutoInferenceReverseDataset, \
    KlueSTSAutoInferenceSignalDataset
from .kornli_dataset import KorNLIAutoInferenceDataset, KorNLIAutoInferenceReverseDataset, \
    KorNLIAutoInferenceSignalDataset

# with-paraphrase dataset
from .qnli_dataset import QNLIAutoParaInferenceDataset, QNLIAutoParaInferenceReverseDataset, \
    QNLIAutoParaInferenceSignalDataset