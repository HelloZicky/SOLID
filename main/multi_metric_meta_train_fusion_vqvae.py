# coding=utf-8
import os
import time
import json
import logging
import math
import argparse
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.multiprocessing as mp
from torch import nn

import model
from util.timer import Timer
from util import args_processing as ap
from util import consts
from util import env
from util import new_metrics
from loader import multi_metric_meta_sequence_dataloader as meta_sequence_dataloader
import numpy as np
from thop import profile

from util import utils
utils.setup_seed(0)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", type=str, help="Kernels configuration for CNN")
    parser.add_argument("--bucket", type=str, default=None, help="Bucket name for external storage")
    parser.add_argument("--dataset", type=str, default="alipay", help="Bucket name for external storage")

    parser.add_argument("--max_steps", type=int, help="Number of iterations before stopping")
    parser.add_argument("--snapshot", type=int, help="Number of iterations to dump model")
    parser.add_argument("--checkpoint_dir", type=str, help="Path of the checkpoint path")
    parser.add_argument("--pretrain_model_path", type=str, help="Path of the checkpoint path")
    parser.add_argument('--pretrain', '-p', action='store_true', help='load pretrained model')

    parser.add_argument("--learning_rate", type=str, default=0.001)
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--max_epoch", type=int,  default=10, help="Max epoch")
    parser.add_argument("--num_loading_workers", type=int, default=4, help="Number of threads for loading")
    parser.add_argument("--model", type=str, help="model type")
    parser.add_argument("--init_checkpoint", type=str, default="", help="Path of the checkpoint path")
    parser.add_argument("--init_step", type=int, default=0, help="Path of the checkpoint path")

    parser.add_argument("--max_gradient_norm", type=float, default=0.)

    parser.add_argument("--arch_config_path", type=str, default=None, help="Path of model configs")
    parser.add_argument("--arch_config", type=str, default=None, help="base64-encoded model configs")

    return parser.parse_known_args()[0]


def predict(predict_item_dataset, predict_cate_dataset, model_obj, device, args, train_epoch, train_step, writer=None):

    model_obj.eval()
    model_obj.to(device)

    timer = Timer()
    log_every = 200

    pred_list = []
    y_list = []
    buffer = []
    user_id_list = []
    for step, batch_data in enumerate(zip(predict_item_dataset, predict_cate_dataset), 1):
        batch_item_data, batch_cate_data = batch_data
        item_features = {
            key: value.to(device)
            for key, value in batch_item_data.items()
            if key not in {consts.FIELD_USER_ID, consts.FIELD_LABEL}
        }
        cate_features = {
            key: value.to(device)
            for key, value in batch_cate_data.items()
            if key not in {consts.FIELD_USER_ID, consts.FIELD_LABEL}
        }
        logits, target_embed, vae_loss = model_obj(item_features, cate_features)
        prob = torch.sigmoid(logits).detach().cpu().numpy()
        y = batch_item_data[consts.FIELD_LABEL].view(-1, 1)
        try:
            overall_auc, _, _, _ = new_metrics.calculate_overall_auc(prob, y)
        except ValueError:
            pass

        user_id_list.extend(np.array(batch_item_data[consts.FIELD_USER_ID].view(-1, 1)))
        pred_list.extend(prob)
        y_list.extend(np.array(y))

        buffer.extend(
            [int(user_id), float(score), float(label)]
            for user_id, score, label
            in zip(
                batch_item_data[consts.FIELD_USER_ID],
                prob,
                batch_item_data[consts.FIELD_LABEL]
            )
        )

        if step % log_every == 0:
            logger.info(
                "train_epoch={}, step={}, overall_auc={:5f}, speed={:2f} steps/s".format(
                    train_epoch, step, overall_auc, log_every / timer.tick(False)
                )
            )

    overall_auc, _, _, _ = new_metrics.calculate_overall_auc(np.array(pred_list), np.array(y_list))
    user_auc = new_metrics.calculate_user_auc(buffer)
    overall_logloss = new_metrics.calculate_overall_logloss(np.array(pred_list), np.array(y_list))
    user_ndcg5, user_hr5 = new_metrics.calculate_user_ndcg_hr(5, buffer)
    user_ndcg10, user_hr10 = new_metrics.calculate_user_ndcg_hr(10, buffer)
    user_ndcg20, user_hr20 = new_metrics.calculate_user_ndcg_hr(20, buffer)

    print("train_epoch={}, train_step={}, overall_auc={:5f}, user_auc={:5f}, overall_logloss={:5f}, "
          "user_ndcg5={:5f}, user_hr5={:5f}, user_ndcg10={:5f}, user_hr10={:5f}, user_ndcg20={:5f}, user_hr20={:5f}".
          format(train_epoch, train_step, overall_auc, user_auc, overall_logloss,
                 user_ndcg5, user_hr5, user_ndcg10, user_hr10, user_ndcg20, user_hr20))
    with open(os.path.join(args.checkpoint_dir, "log_test.txt"), "a") as writer:
        print("train_epoch={}, train_step={}, overall_auc={:5f}, user_auc={:5f}, overall_logloss={:5f}, "
              "user_ndcg5={:5f}, user_hr5={:5f}, user_ndcg10={:5f}, user_hr10={:5f}, user_ndcg20={:5f}, user_hr20={:5f}".
              format(train_epoch, train_step, overall_auc, user_auc, overall_logloss,
                     user_ndcg5, user_hr5, user_ndcg10, user_hr10, user_ndcg20, user_hr20), file=writer)

    return overall_auc, user_auc, overall_logloss, user_ndcg5, user_hr5, user_ndcg10, user_hr10, user_ndcg20, user_hr20


def train(train_item_dataset, train_cate_dataset, model_obj, device, args, pred_dataloader1, pred_dataloader2, pretrain_category_weight):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model_obj.parameters(),
        lr=float(args.learning_rate)
    )
    model_obj.train()
    model_obj.to(device)

    print(model_obj)

    logger.info("Start training...")
    timer = Timer()
    log_every = 200
    max_step = 0
    best_auc = 0
    best_auc_ckpt_path = os.path.join(args.checkpoint_dir, "best_auc" + ".pkl")
    for epoch in range(1, args.max_epoch + 1):
        for step, batch_data in enumerate(zip(train_item_dataset, train_cate_dataset), 1):
            batch_item_data, batch_cate_data = batch_data
            item_features = {
                key: value.to(device)
                for key, value in batch_item_data.items()
                if key not in {consts.FIELD_USER_ID, consts.FIELD_LABEL}
            }
            cate_features = {
                key: value.to(device)
                for key, value in batch_cate_data.items()
                if key not in {consts.FIELD_USER_ID, consts.FIELD_LABEL}
            }
            logits, target_embed, vae_loss = model_obj(item_features, cate_features, norm=0.01) # norm=0.01
            
            bce_loss = criterion(logits, batch_item_data[consts.FIELD_LABEL].view(-1, 1).to(device))
            loss = bce_loss + 0.1 * vae_loss # lambda=0.1
            pred, y = torch.sigmoid(logits), batch_item_data[consts.FIELD_LABEL].view(-1, 1)
            auc, _, _, _ = new_metrics.calculate_overall_auc(np.array(pred.detach().cpu()), np.array(y))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % log_every == 0:
                logger.info(
                    "epoch={}, step={}, loss={:5f}, auc={:5f}, speed={:2f} steps/s, ratio={:5f}".format(
                        epoch, step, float(loss.item()), auc, log_every / timer.tick(False), vae_loss.item() / bce_loss.item()
                    )
                )
            max_step = step

        pred_overall_auc, pred_user_auc, pred_overall_logloss, pred_user_ndcg5, pred_user_hr5, \
        pred_user_ndcg10, pred_user_hr10, pred_user_ndcg20, pred_user_hr20 = predict(
            predict_item_dataset=pred_dataloader1,
            predict_cate_dataset=pred_dataloader2,
            model_obj=model_obj,
            device=device,
            args=args,
            train_epoch=epoch,
            train_step=epoch * max_step,
        )
        logger.info("dump checkpoint for epoch {}".format(epoch))
        model_obj.train()
        if pred_user_auc > best_auc:
            best_auc = pred_user_auc
            torch.save(model_obj, best_auc_ckpt_path)


def main_worker(_):
    args = parse_args()
    ap.print_arguments(args)

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    model_meta = model.get_model_meta(args.model)  # type: model.ModelMeta

    device = env.get_device()
    
    # load pretrained weight
    if args.pretrain:
        ckpt = torch.load(args.pretrain_model_path, map_location='cpu')
        ckpt.to(device)
        pretrain_category_weight = ckpt.state_dict()['_id_encoder._embedding_matrix.weight']
        print(pretrain_category_weight)
    else:
        pretrain_category_weight = None

    model_conf, raw_model_conf = ap.parse_arch_config_from_args(model_meta, args)  # type: dict

    model_obj = model_meta.model_builder(model_conf=model_conf, pretrain_category_weight=pretrain_category_weight)  # type: torch.nn.module
    print("=" * 100)
    for name, parms in model_obj.named_parameters():
        print(name)
    print("=" * 100)

    worker_id = worker_count = 8
    train_file, test_file = args.dataset.split(";")
    train_file = train_file.split(",")
    test_file = test_file.split(",")

    args.num_loading_workers = 1

    train_dataloader1 = meta_sequence_dataloader.MetaSequenceDataLoader(
        table_name=train_file[0],
        slice_id=0,
        slice_count=args.num_loading_workers,
        is_train=True
    )
    train_dataloader1 = torch.utils.data.DataLoader(
        train_dataloader1,
        batch_size=args.batch_size,
        num_workers=args.num_loading_workers,
        pin_memory=True,
        collate_fn=train_dataloader1.batchify,
        drop_last=False
    )
    train_dataloader2 = meta_sequence_dataloader.MetaSequenceDataLoader(
        table_name=train_file[1],
        slice_id=0,
        slice_count=args.num_loading_workers,
        is_train=True
    )
    train_dataloader2 = torch.utils.data.DataLoader(
        train_dataloader2,
        batch_size=args.batch_size,
        num_workers=args.num_loading_workers,
        pin_memory=True,
        collate_fn=train_dataloader2.batchify,
        drop_last=False
    )

    # Setup up data loader
    pred_dataloader1 = meta_sequence_dataloader.MetaSequenceDataLoader(
        table_name=test_file[0],
        slice_id=args.num_loading_workers * worker_id,
        slice_count=args.num_loading_workers * worker_count,
        is_train=False
    )
    pred_dataloader1 = torch.utils.data.DataLoader(
        pred_dataloader1,
        batch_size=args.batch_size,
        num_workers=args.num_loading_workers,
        pin_memory=True,
        collate_fn=pred_dataloader1.batchify,
        drop_last=False
    )
    pred_dataloader2 = meta_sequence_dataloader.MetaSequenceDataLoader(
        table_name=test_file[1],
        slice_id=args.num_loading_workers * worker_id,
        slice_count=args.num_loading_workers * worker_count,
        is_train=False
    )
    pred_dataloader2 = torch.utils.data.DataLoader(
        pred_dataloader2,
        batch_size=args.batch_size,
        num_workers=args.num_loading_workers,
        pin_memory=True,
        collate_fn=pred_dataloader2.batchify,
        drop_last=False
    )

    # Setup training
    train(
        train_item_dataset=train_dataloader1,
        train_cate_dataset=train_dataloader2,
        model_obj=model_obj,
        device=device,
        args=args,
        pred_dataloader1=pred_dataloader1,
        pred_dataloader2=pred_dataloader2,
        pretrain_category_weight=pretrain_category_weight
    )


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    mp.spawn(main_worker, nprocs=1)

