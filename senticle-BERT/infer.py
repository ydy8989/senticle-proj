from transformers import AutoTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, BertConfig, \
    BertTokenizer, AutoConfig, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from load_data import *
import pandas as pd
import torch
import pickle as pickle
import numpy as np
import argparse
from train import YamlConfigManager
# import argparse
from importlib import import_module
import glob
from prettyprinter import cpprint


def inference(model, tokenized_sent, device):
    dataloader = DataLoader(tokenized_sent, batch_size=40, shuffle=False)
    model.eval()
    output_pred = []
    logits_list = []
    for i, data in enumerate(dataloader):
        with torch.no_grad():
            outputs = model(
                input_ids=data['input_ids'].to(device),
                attention_mask=data['attention_mask'].to(device),
                # token_type_ids=data['token_type_ids'].to(device)
            )
        # print(outputs[0])
        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        logits_list.append(logits)
        result = np.argmax(logits, axis=-1)

        output_pred.append(result)

    return logits_list#np.array(output_pred).flatten()


def load_test_dataset(dataset_dir, tokenizer):
    test_dataset = load_data(dataset_dir)
    test_label = test_dataset['label'].values
    # tokenizing dataset
    tokenized_test = tokenized_dataset(test_dataset, tokenizer)
    return tokenized_test, test_label


def main(args,cfg):

    """
      주어진 dataset tsv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # load tokenizer
    #   TOK_NAME = "bert-base-multilingual-cased"
    TOK_NAME = cfg.values.model_name#args.pretrained_model
    tokenizer = AutoTokenizer.from_pretrained(TOK_NAME)
    model_config = AutoConfig.from_pretrained(TOK_NAME)
    model_config.num_labels = 42
    best_models = sorted(glob.glob(os.path.join(args.model_dir,'best.pt')))
    # best_models2 = sorted(glob.glob(os.path.join(args.model_dir2, '*/best.pt')))
    print(best_models)
    logits_lists = []
    for idx, fold_best_model in enumerate(best_models):#+best_models2
        cpprint('=' * 15 + f'{idx+1}-Fold Inference' + '=' * 15)
        model = AutoModelForSequenceClassification.from_pretrained(TOK_NAME, config=model_config)
        model.load_state_dict(torch.load(fold_best_model))
        model.to(device)

        print(f"Loaded pretrained weights from {fold_best_model}", end="\t")

        # load test datset
        test_dataset_dir = "../input/data/test/test.tsv"
        test_dataset, test_label = load_test_dataset(test_dataset_dir, tokenizer)
        test_dataset = RE_Dataset(test_dataset, test_label)

        # predict answer
        logits_list = inference(model, test_dataset, device)
        logits_lists.append(logits_list)

    # fold 하나만 inference
    if len(logits_lists)==1:
        [fold1] = logits_lists  #
        val_out_pred = []
        for val_iter in range(len(fold1)):
            val_logit = fold1[val_iter]
            result = np.argmax(val_logit, axis=-1)
            val_out_pred.append(result)

    # fold 5개 전체 inference
    elif len(logits_lists)==5:
        [fold1, fold2, fold3, fold4, fold5] = logits_lists  #
        val_out_pred = []
        for val_iter in range(len(fold1)):
            val_logit = fold1[val_iter]+fold2[val_iter]+fold3[val_iter]+fold4[val_iter]+fold5[val_iter]
            result = np.argmax(val_logit, axis=-1)
            val_out_pred.append(result)

    elif len(logits_lists) == 10:
        [a, b, c, d, e, f, g, h, i, j] = logits_lists  #
        val_out_pred = []
        for val_iter in range(len(a)):
            val_logit = a[val_iter] + b[val_iter] + c[val_iter] + d[val_iter] + e[val_iter] + f[val_iter] + g[val_iter]+h[val_iter]+i[val_iter]+j[val_iter]
            result = np.argmax(val_logit, axis=-1)
            val_out_pred.append(result)

    kfold_answer = np.array(val_out_pred).flatten()
    output = pd.DataFrame(kfold_answer, columns=['pred'])
    output.to_csv(f'./prediction/submission_{args.model_dir.split("/")[-1]}_lastlast.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # model dir
    parser.add_argument('--model_dir', type=str, default="./results/roberta")
    # parser.add_argument('--model_dir2', type=str, default="./results/roberta")
    parser.add_argument('--config_file_path', type=str, default='./config.yml')
    parser.add_argument('--config', type=str, default='roberta')
    args = parser.parse_args()
    cfg = YamlConfigManager(args.config_file_path, args.config)
    print(args)
    print(f'{args.model_dir.split("/")[-1]}')

    main(args, cfg)
