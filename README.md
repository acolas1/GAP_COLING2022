# GAP: A Graph-aware Language Model Framework for Knowledge Graph-to-Text Generation
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/gap-a-graph-aware-language-model-framework/kg-to-text-generation-on-webnlg-2-0)](https://paperswithcode.com/sota/kg-to-text-generation-on-webnlg-2-0?p=gap-a-graph-aware-language-model-framework)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/gap-a-graph-aware-language-model-framework/kg-to-text-generation-on-eventnarrative)](https://paperswithcode.com/sota/kg-to-text-generation-on-eventnarrative?p=gap-a-graph-aware-language-model-framework)

## Paper
Accepted as a Main Conference Long paper at COLING 2022. Paper can be found [here](https://aclanthology.org/2022.coling-1.506.pdf).

**Abstract**: Recent improvements in KG-to-text generation are due to additional auxiliary pre-training tasks designed to give the fine-tune task a boost in performance. These tasks require extensive computational resources while only suggesting marginal improvements. Here, we demonstrate that by fusing graph-aware elements into existing pre-trained language models, we are able to outperform state-of-the-art models and close the gap imposed by additional pre-training tasks. We do so by proposing a mask structure to capture neighborhood information and a novel type encoder that adds a bias to the graph-attention weights depending on the connection type. Experiments on two KG-to-text benchmark datasets show our models are competitive while involving fewer parameters and no additional pre-training tasks. By formulating the problem as a framework, we can interchange the various proposed components and begin interpreting KG-to-text generative models based on the topological and type information found in a graph. 

# Dependencies
* Python 3.7
* PyTorch 1.9.0
* PyTorch Scatter 2.0.9 
* Transformers 3.0.0
* Matplotlib
* scikit-image

#### Installation with CUDA 11.1
`pip install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html`

`pip install torch-scatter -f https://data.pyg.org/whl/1.9.0+cu111.html`

## Directories
- `data/`: create directory to store data in here
- `pretrained_LM/`: create directory to store pre-trained language model
- `out/`: stores finetuned model and log files

## Currently Supported Data
- [eventNarrative](https://www.kaggle.com/acolas1/eventnarration)
- [webnlg](https://drive.google.com/drive/folders/1Jx3Cz7t0hiNdtlBWUdPjhtLwPOH3LtzC?usp=share_link)

**Note**: For EventNarrative please add the following hyper-parameters to the finetune and infer files:
* --max_node_length 60
* ----length_penalty 5.0
* --dataset eventNarrative

## Pre-trained Language Model
Please download the pre-trained_LM bart-base from Huggingface found [here](https://huggingface.co/facebook/bart-base) unzip and move it to the `pretrained_LM/` folder

## Usage
**Fine-tune**

```shell
bash finetune.sh
```

**Inference**

```shell
bash infer.sh
```

**Evaluation**

To compute the METEOR scores, please download the required [data](https://github.com/xinyadu/nqg/blob/master/qgevalcap/meteor/data/paraphrase-en.gz) and put it under `eval_webnlg/pycocoevalcap/meteor/data/` and `eval_wqpq/meteor/data/`.

For a fair comparison with existing works, we use the evaluation scripts of [KGPT](https://github.com/wenhuchen/KGPT): 

```shell
cd eval_webnlg
python measure_score.py ${reference_path} ${generated_path}
```

where `reference_path` is the target dataset and `generated_path` is the data which is generated from the inference script.

**Important Flags**:
  * `--output_dir` : where model and results our output
  * `--model_path` : path for loading pre-trained checkpoint
  * `--tokenizer_path` : path for loading pre-trained tokenizer
  * `--train_file` : path to training file
  * `--predict_file`: path to predict file
  * Masking flags: 
    * `--entity_entity`
    * `--entity_relation`
    * `--relation_entity`
    * `--relation_relation`
  * `--type_encoding`: Flag for turning on type encoding matrix
  
## Citation
Please cite our paper if using this repository:
```
@inproceedings{colas-etal-2022-gap,
    title = "{GAP}: A Graph-aware Language Model Framework for Knowledge Graph-to-Text Generation",
    author = "Colas, Anthony  and
      Alvandipour, Mehrdad  and
      Wang, Daisy Zhe",
    booktitle = "Proceedings of the 29th International Conference on Computational Linguistics",
    month = oct,
    year = "2022",
    address = "Gyeongju, Republic of Korea",
    publisher = "International Committee on Computational Linguistics",
    url = "https://aclanthology.org/2022.coling-1.506",
    pages = "5755--5769"
}
```

## References
We refer to the following repositories for parts of our code:
[Transformers](https://github.com/huggingface/transformers), [bart-closed-book-qa](https://github.com/shmsw25/bart-closed-book-qa), and [JointGT](https://github.com/thu-coai/JointGT). 

