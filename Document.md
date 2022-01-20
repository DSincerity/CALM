### 1. Related Papers
#### Consistency
- *Negated and Misprimed Probes for Pretrained Language Models:
Birds Can Talk, But Cannot Fly*, Kassner and Schütze (2020), [[DOI]](https://www.aclweb.org/anthology/2020.acl-main.698/)
- *On the Systematicity of Probing Contextualized Word Representations:
The Case of Hypernymy in BERT*, Ravichander et al., (2020) [[DOI]](https://www.aclweb.org/anthology/2020.starsem-1.10/)
- *Measuring and Improving Consistency in Pretrained Language Models*, Elazer et al., (2021), [[DOI]](https://arxiv.org/abs/2102.01017)
- *Make Up Your Mind! Adversarial Generation of Inconsistent Natural Language Explanations*, Camburu et al., (2020) [[DOI]](https://arxiv.org/abs/1910.03065)
- *Are Red Roses Red? Evaluating Consistency of Question-Answering Models*, Ribeiro et al., (2020), [[DOI]](https://www.aclweb.org/anthology/P19-1621/)
- *Don't Say That! Making Inconsistent Dialogue Unlikely with Unlikelihood Training*, Li et al., (2020), [[DOI]](https://www.aclweb.org/anthology/2020.acl-main.428/)
- *Enriching a Model's Notion of Belief using a Persistent Memory*, Kassner et al., (2021), [[DOI]](https://arxiv.org/abs/2104.08401)
- *Consistency of a Recurrent Language Model With Respect to Incomplete Decoding*, Welleck et al., (2020), [[DOI]](https://arxiv.org/abs/2002.02492)
- *Logic-Guided Data Augmentation and Regularization for Consistent Question Answering*, Asai and Hajishirzi, (2020), [[DOI]](https://www.aclweb.org/anthology/2020.acl-main.499/)
- *Evaluating the factual consistency of abstractive text summarization.*, Kryscinski et al., (2020), [[DOI]](https://www.aclweb.org/anthology/info/corrections/)


#### Prompt Engineering
- *GPT Understands, Too*, Liu et al., (2021) [[DOI]](https://arxiv.org/pdf/2103.10385v1.pdf)
- *How Can We Know What Language Models Know?*, Jiang et al., (2020), [[DOI]](https://www.aclweb.org/anthology/2020.tacl-1.28/)

### 2. Datasets (Tasks)
| Dataset | Task                                   | Remark |
|---------|----------------------------------------|--------|
| RTE     | Recognizing Textual Entailment         | GLUE   |
| MNLI    | Multi-Genre Natural Language Inference | GLUE   |
| QQP     | Quora Question Pairs                   | GLUE   |
| MRPC    | Microsoft Research Paraphrate Corpus   | GLUE   |
| QNLI    | Question Natural Language Inference    | GLUE   |
| COPA    | Choice Of Plausible Alternatives       | SUPER GLUE  |


### 3. Paper Structure (Draft)
#### 1) Introduction
- PLM 등장, PLM이 언어를 이해한다는 주장들이 나옴
- PLM의 언어 이해 능력에 의문을 제기하는 연구들
- Understand의 정의: 의미를 인지하는 것 -> 언어를 이해하면 consistent한 행동을 보여야 함 (consistency 정의 cite)
- PLM이 consistent 하지 않음을 보이는 몇가지 연구: prompt engineering / adversarial attack on PLM
- 본 연구의 의의: 
    - Consistency를 측정하는 task 제안: 기존 연구와 다른점은 완전히 똑같은 의미를 가진 input에 대해 test. Extremely conservative.
    - PLM들이 consistent 하지 않음을 보임 (모델, 언어에 상관 없) + 사람이라면 전혀 하지 않을 행동 (human eval)이
    - Paraphrase data: Consistency 향상에 기여

#### 2) Related works
- Analysis on Consistency (직접적 관련)
- Adversarial attack on PLM (비슷한 맥락 but 차이점 언급)

#### 3) Experiments

#### 4) Results and Analysis
1. English
- High consistency for paraphrase identification task
- Encoding models: robust to SIGNAL changes but show weakness in REVERSE cases
- Text-to-Text: better in REVERSE than encoding models. But fails in SIGNAL cases (examples)
- Effect of model size

2. Korean
- TBD


#### 5) Conclusion