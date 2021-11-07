import argparse
import os
import random
import re
import warnings
from glob import glob
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from easydict import EasyDict
from prettyprinter import cpprint
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import *

from kobert_tokenization import KoBertTokenizer
from load_data import *
from loss import *

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')


# 실험을 위한 random seed 고정
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


# 실험을 위한 모델 저장 파일명 자동 변경
def increment_output_dir(output_path, exist_ok=False):
    path = Path(output_path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


# config.yml 파싱 위한 config매니저
class YamlConfigManager:
    def __init__(self, config_file_path, config_name):
        super().__init__()
        self.values = EasyDict()
        if config_file_path:
            self.config_file_path = config_file_path
            self.config_name = config_name
            self.reload()

    def reload(self):
        self.clear()
        if self.config_file_path:
            with open(self.config_file_path, 'r') as f:
                self.values.update(yaml.safe_load(f)[self.config_name])

    def clear(self):
        self.values.clear()

    def update(self, yml_dict):
        for (k1, v1) in yml_dict.items():
            if isinstance(v1, dict):
                for (k2, v2) in v1.items():
                    if isinstance(v2, dict):
                        for (k3, v3) in v2.items():
                            self.values[k1][k2][k3] = v3
                    else:
                        self.values[k1][k2] = v2
            else:
                self.values[k1] = v1

    def export(self, save_file_path):
        if save_file_path:
            with open(save_file_path, 'w') as f:
                yaml.dump(dict(self.values), f)


def train(cfg, data_path):
    train_df = pd.read_csv(data_path)
    train_label = train_df.label.values
    train_dataset = train_df.text

    # hyperparameter
    SEED = cfg.values.seed
    seed_everything(SEED)
    MODEL_NAME = cfg.values.model_name
    USE_KFOLD = cfg.values.val_args.use_kfold
    log_interval = cfg.values.train_args.log_interval
    weight_decay = cfg.values.train_args.weight_decay
    tr_batch_size = cfg.values.train_args.train_batch_size
    val_batch_size = cfg.values.train_args.eval_batch_size
    max_seqlen = cfg.values.train_args.max_seqlen
    epochs = cfg.values.train_args.num_epochs
    loss_type = cfg.values.train_args.loss_fn
    lr_decay_step = 1  # stepLR parameter
    steplr_gamma = cfg.values.train_args.steplr_gamma
    opti = cfg.values.train_args.optimizer
    scheduler_type = cfg.values.train_args.scheduler_type
    label_smoothing_factor = cfg.values.train_args.label_smoothing_factor
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'device is "{device}"')
    print('MODEL_NAME:', MODEL_NAME)

    ##################################################################################
    #                              huggingface config
    ##################################################################################
    # transformer의 버전에 따라 종종 electra 모델의 autoconfig 사용시 버그 이슈 있어 autoconfig 사용하지 않음
    if 'koelectra' in MODEL_NAME:
        model_config = ElectraConfig.from_pretrained(MODEL_NAME)
    elif 'bigbird' in MODEL_NAME:
        model_config = AutoConfig.from_pretrained(MODEL_NAME)
        model_config.block_size = 16
        model_config.num_random_blocks = 2
    else:
        model_config = AutoConfig.from_pretrained(MODEL_NAME)

    model_config.num_labels = 2

    ##################################################################################
    #                                    Tokenizer
    ##################################################################################
    if MODEL_NAME == 'KoBertTokenizer':
        tokenizer = KoBertTokenizer.from_pretrained(MODEL_NAME)
    # electra 모델의 autotokenizer 사용시 버그 이슈 있어 autotokenizer 사용하지 않음
    elif 'koelectra' in MODEL_NAME:
        tokenizer = ElectraTokenizer.from_pretrained(MODEL_NAME)
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    class_weights = compute_class_weight('balanced', np.unique(train_label), train_label)
    weights = torch.tensor(class_weights, dtype=torch.float)
    weights = weights.to(device)

    ##################################################################################
    #                                       LOSS
    ##################################################################################
    if loss_type == 'custom':  # F1 + Cross_entropy
        criterion = CustomLoss()
    elif loss_type == 'labelsmooth':
        criterion = LabelSmoothingLoss(smoothing=label_smoothing_factor)
    elif loss_type == 'CEloss':
        criterion = nn.CrossEntropyLoss(weight=weights)
    elif loss_type == 'focal':
        criterion = FocalLoss()

    ##################################################################################
    #                                 Training Start
    ##################################################################################
    if USE_KFOLD:
        # Data Imbalance 완화를 위한 Stratified kfold 적용
        kfold = StratifiedKFold(n_splits=5)
        k = 1
        save_dir = increment_output_dir(cfg.values.train_args.save_dir)
        print(save_dir)
        for idx, splits in enumerate(kfold.split(train_dataset, train_label)):
            trind, valind = splits
            tr_label = train_label[trind]
            val_label = train_label[valind]
            print(valind)
            print(f'-----fold_{idx}의 train/val의 클래스 비율-------')
            # for i in range(model_config.num_labels):
            #     print(f'{i}의 개수 : {tr_label.tolist().count(i)/len(tr_label):4.2%} / {val_label.tolist().count(i)/len(val_label):4.2%}')
            tr_dataset = train_dataset.iloc[trind]
            val_dataset = train_dataset.iloc[valind]
            tokenized_train = tokenized_dataset(tr_dataset, tokenizer, max_seqlen)
            tokenized_dev = tokenized_dataset(val_dataset, tokenizer, max_seqlen)
            #             print(tokenized_train)
            Zum_train_dataset = ZumDataset(tokenized_train, tr_label)
            Zum_dev_dataset = ZumDataset(tokenized_dev, val_label)

            train_loader = DataLoader(Zum_train_dataset, batch_size=tr_batch_size, shuffle=True)
            val_loader = DataLoader(Zum_dev_dataset, batch_size=val_batch_size, shuffle=False)

            ##################################################################################
            #                                      Model
            ##################################################################################
            if 'koelectra' in MODEL_NAME:
                model = ElectraForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
            else:
                model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
            model.to(device)
            model_dir = save_dir + f'/{idx + 1}fold'
            print(model_dir)

            ##################################################################################
            #                              Optimizer and Scheduler
            ##################################################################################
            if scheduler_type == 'stepLr':
                opt_module = getattr(import_module("torch.optim"), opti)
                optimizer = opt_module(
                    filter(lambda p: p.requires_grad,
                           model.parameters()),
                    lr=cfg.values.train_args.lr,
                    weight_decay=weight_decay
                )
                scheduler = StepLR(optimizer, lr_decay_step, gamma=steplr_gamma)  # 794) #gamma : 20epoch => lr x 0.01

            elif scheduler_type == 'cycleLR':
                opt_module = getattr(import_module("torch.optim"), opti)
                optimizer = opt_module(
                    filter(lambda p: p.requires_grad,
                           model.parameters()),
                    lr=cfg.values.train_args.lr,  # 5e-6,
                    weight_decay=weight_decay
                )
                scheduler = CyclicLR(optimizer,
                                     base_lr=0.000000001,
                                     max_lr=cfg.values.train_args.lr,
                                     step_size_up=1,
                                     step_size_down=4,
                                     mode='triangular',
                                     cycle_momentum=False)

            # Tensorboard
            logger = SummaryWriter(log_dir=model_dir)

            best_val_acc = 0
            best_val_loss = np.inf

            ##################################################################################
            #                              Training Loop
            ##################################################################################
            for epoch in range(epochs):
                model.train()
                loss_value = 0
                matches = 0
                for idx, batch in enumerate(train_loader):
                    optimizer.zero_grad()

                    input_ids = batch['input_ids'].to(device)
                    token_type_ids = batch['token_type_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)

                    outputs = model(input_ids,
                                    token_type_ids=token_type_ids,
                                    attention_mask=attention_mask,
                                    labels=labels)

                    loss = criterion(outputs[1], labels)
                    loss_value += loss.item()
                    preds = torch.argmax(F.log_softmax(outputs[1], dim=1), dim=-1)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    matches += (preds == labels).sum().item()
                    if (idx + 1) % log_interval == 0:
                        train_loss = loss_value / log_interval
                        train_acc = matches / tr_batch_size / log_interval
                        current_lr = get_lr(optimizer)
                        print(
                            f"Epoch[{epoch}/{epochs}]({idx + 1}/{len(train_loader)}) || "
                            f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr:.3}"
                        )
                        logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
                        logger.add_scalar("Train/accuracy", train_acc, epoch * len(train_loader) + idx)
                        logger.add_scalar("Train/lr", current_lr, epoch * len(train_loader) + idx)
                        loss_value = 0
                        matches = 0
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                model.eval()
                with torch.no_grad():
                    print("Calculating validation results...")
                    val_loss_items = []
                    val_acc_items = []
                    for idx, val_batch in enumerate(val_loader):
                        input_ids = val_batch['input_ids'].to(device)
                        token_type_ids = val_batch['token_type_ids'].to(device)
                        attention_mask = val_batch['attention_mask'].to(device)
                        labels = val_batch['labels'].to(device)

                        outputs = model(input_ids,
                                        token_type_ids=token_type_ids,
                                        attention_mask=attention_mask,
                                        labels=labels)

                        preds = torch.argmax(F.log_softmax(outputs[1], dim=1), dim=-1)
                        loss_item = outputs[0].item()
                        correct = preds.eq(labels)
                        acc_item = correct.sum().item()

                        val_loss_items.append(loss_item)
                        val_acc_items.append(acc_item)

                    val_loss = np.sum(val_loss_items) / len(val_loader)
                    val_acc = np.sum(val_acc_items) / len(val_label)
                    best_val_loss = min(best_val_loss, val_loss)

                    if val_acc > best_val_acc:
                        print(f"New best model for val accuracy : {val_acc:4.2%}! saving the best model..")
                        torch.save(model.state_dict(), f"./{model_dir}/best.pt")
                        best_val_acc = val_acc
                    torch.save(model.state_dict(), f"./{model_dir}/last.pt")
                    print(
                        f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                        f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
                    )
                    logger.add_scalar("Val/loss", val_loss, epoch)
                    logger.add_scalar("Val/accuracy", val_acc, epoch)
                    print()
            with open(f"./{model_dir}/config.yaml", 'w') as file:
                documents = yaml.dump(cfg.values, file)

            k += 1
            if cfg.values.val_args.fold_break:
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file_path', type=str, default='./config.yml')
    parser.add_argument('--config', type=str, default='bigbird')
    parser.add_argument('--data_path', type=str, default='../data/pre_005930.csv')
    args = parser.parse_args(args=[])

    cfg = YamlConfigManager(args.config_file_path, args.config)
    cpprint(cfg.values, sort_dict_keys=False)

    # 학습 시작
    train(cfg, args.data_path)
