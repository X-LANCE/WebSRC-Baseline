# WebSRC

This repository contains the full pipeline to train and evaluate the baseline models in the paper [WebSRC: A Dataset for Web-Based Structural Reading Comprehension](https://arxiv.org/abs/2101.09465) on the WebSRC dataset. The dataset and leaderboard can be found [here](https://x-lance.github.io/WebSRC/).

## Latest Experiment Result
All results are listed in the format of *dev/test*


method | exact | f1 | pos 
-------| --- | --- |---
bert-tplm | 52.12/39.28 | 61.57/49.49 | 79.74/67.68
bert-hplm | 61.51/52.61 | **67.04**/59.88 | 82.97/76.13
bert-vplm | **62.07**/**52.84** | 66.66/**60.80** | **83.64**/**76.39**
|||
electra-tplm | 61.67/56.32 | 69.85/72.35 | 84.15/79.18
electra-hplm | 70.12/66.29 | 74.14/72.71 | 86.33/83.17
electra-vplm | **73.22**/**68.07** | **76.16**/**75.25** | **87.06**/**84.96**

## Requirements

The required python packages is listed in "requirements.txt". You can install them by
```commandline
pip install -r requirements.txt
```
or
```commandline
conda install --file requirements.txt
```

## Data Format Description

The dataset for each website will be stored in `dataset.csv` in the directory `{domain-name}/{website-number}`. The corresponding raw data (including HTML files, screenshots, bounding box coordinates, and page names and urls) is stored in the `processed_data` folder in the same directory.

In `dataset.csv`, each row corresponding to one question-answer data point except the header. The meanings of each column are as follows:
* `question`: a string, the question of this question-answer data point.
* `id`: a unique id for this question-answer data point. Each `id` has a length 14, the first two characters are the domain indicator, the following two number is the website name. The corresponding page id can be extracted by `id[2:9]`, for example, id "sp160000100001" means this line is created from the *sport* domain, website *16*, and the corresponding page is `1600001.html`.
* `element_id`: an integer, the tag id (corresponding to the tag's `tid` attribute in the HTML files) of the deepest tag in the DOM tree which contain all the answer. For yes/no question, there is no tag associated with the answer, so the `element_id` is -1.
* `answer_start`: an integer, the char offset of the answer from the start of the content of the tag specified by `element_id`. Note that before counting this number, we first eliminate all the inner tags in the specified tag and replace all the consecutive whitespaces with one space. For yes/no questions, `answer_start` is 1 for answer "yes" and 0 for answer "no".
* `answer`: a string, the answer of this question-answer data point.

## Dataset Testing

The sample of the WebSRC dataset is shown in the `data` directory. The full dataset can be downloaded [here](https://x-lance.github.io/WebSRC/).

To test the correctness of the dataset format, run the following line to test the whole dataset:

```commandline
bash ./test/test_all.sh ./data
```
Note that you should first put the whole dataset in the `data/` folder. 
To test one specific website, for example `au08`, run:

```commandline
cd test
pytest --domain=auto --data=../data/auto/08 --website=08
```

Details about how to parse html files and locate answers in html please refer to `test/test_label.py`.

## Data Pre-processing

To run the baseline models, we need to first convert the dataset into SQuAD-style json files. To achieve this, switch into the `src` directory and run
```commandline
python dataset_generation.py --root_dir ../data --version websrc1.0
```
The resulting SQuAD-style json files will be placed in the `data` folder.

Furthermore, to run the V-PLM model, the processed cnn features for each tag in the pages can be downloaded from [Amazon s3](https://websrc-data.s3.amazonaws.com/visual-features.xz) or [Baidu Netdisk](https://pan.baidu.com/s/1_KeVmazOdCrU33nhiKUyRg) with the Extraction Code: 'pi5s'.

[comment]: <> (lack download links)

## Training

After pre-processing the data, the baseline models can be trained. To do so, stay in the `src` directory and run the `train.sh` files in the directory `./script/{backbone-PLM-model-name}/{method-name}/`. For example, to train the H-PLM model with BERT backbone, run the following command under the `src` folder:
```commandline
bash ./script/BERT/H-PLM/train.sh
```

## Evaluation

The `eval.sh` files which can evaluate all the saved checkpoints on the development set are placed in the same folder as the `train.sh` files for the same method. For example, to evaluate all the checkpoints saving by the previous command, run the following command under the `src` folder:
```commandline
bash ./script/BERT/H-PLM/eval.sh
```

## Obtain Test Result

The test set without label is public now, available [here][https://websrc-data.s3.amazonaws.com/release_testset.zip], the format of test set is the same with training set (please refer to the Data Format Description section). For test set evaluation, please send your prediction files to  zhao_mengxin@sjtu.edu.cn and chenlusz@sjtu.edu.cn with title "WebSRC Test: \<your model name\>+\<your institution\>". For evaluation, the prediction files should contain two files:

```jsonc
// prediction.json
// A json format file, keys are ids and values are the predicted answers (string).
{
  "sp160000100001": "predicted answer",
  "sp160000100002": "...",
  //...
}

// tag_prediction.json
// A json format file, keys are ids and values are the predicted tag tid (int)
{
  "sp160000100001": -1,
  "sp160000100002": -1,
  //...
}
```

We encourage to submit results from **at least three runs with different random seeds** to reduce the uncertainty of the experiments. Please place prediction files for each runs in different directory and submit a zipped file. The averaged test result will be sent by email.



## Reference

If you use any source codes or datasets included in this repository in your work, please cite the corresponding papers. The bibtex are listed below:
```text
@inproceedings{chen-etal-2021-websrc,
    title = "{W}eb{SRC}: A Dataset for Web-Based Structural Reading Comprehension",
    author = "Chen, Xingyu  and
      Zhao, Zihan  and
      Chen, Lu  and
      Ji, JiaBao  and
      Zhang, Danyang  and
      Luo, Ao  and
      Xiong, Yuxuan  and
      Yu, Kai",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.343",
    pages = "4173--4185",
    abstract = "Web search is an essential way for humans to obtain information, but it{'}s still a great challenge for machines to understand the contents of web pages. In this paper, we introduce the task of web-based structural reading comprehension. Given a web page and a question about it, the task is to find an answer from the web page. This task requires a system not only to understand the semantics of texts but also the structure of the web page. Moreover, we proposed WebSRC, a novel Web-based Structural Reading Comprehension dataset. WebSRC consists of 400K question-answer pairs, which are collected from 6.4K web pages with corresponding HTML source code, screenshots, and metadata. Each question in WebSRC requires a certain structural understanding of a web page to answer, and the answer is either a text span on the web page or yes/no. We evaluate various strong baselines on our dataset to show the difficulty of our task. We also investigate the usefulness of structural information and visual features. Our dataset and baselines have been publicly available.",
}
```
