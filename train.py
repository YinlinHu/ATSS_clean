import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, sampler
from tqdm import tqdm

from argument import get_args
from backbone import vovnet39, vovnet57, resnet18, resnet50, resnet101
from dataset import COCODataset, collate_fn
from model import ATSS
from transform import preset_transform
from evaluate import evaluate
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    DistributedSampler,
    all_gather,
)

from tensorboardX import SummaryWriter

def accumulate_predictions(predictions):
    all_predictions = all_gather(predictions)

    if get_rank() != 0:
        return

    predictions = {}

    for p in all_predictions:
        predictions.update(p)

    ids = list(sorted(predictions.keys()))

    if len(ids) != ids[-1] + 1:
        print('Evaluation results is not contiguous')

    predictions = [predictions[i] for i in ids]

    return predictions


@torch.no_grad()
def valid(args, epoch, loader, dataset, model, device, logger=None):
    if args.distributed:
        model = model.module

    torch.cuda.empty_cache()

    model.eval()

    if get_rank() == 0:
        pbar = tqdm(enumerate(loader), total=len(loader), dynamic_ncols=True)
    else:
        pbar = enumerate(loader)

    preds = {}

    for idx, (images, targets, ids) in pbar:
        model.zero_grad()

        images = images.to(device)
        targets = [target.to(device) for target in targets]

        pred, _ = model(images)

        pred = [p.to('cpu') for p in pred]

        preds.update({id: p for id, p in zip(ids, pred)})

    preds = accumulate_predictions(preds)

    if get_rank() != 0:
        return
        
    evl_res = evaluate(dataset, preds)

    # writing log to tensorboard
    if logger:
        log_group_name = "validation"
        box_result = evl_res['bbox']
        logger.add_scalar(log_group_name + '/AP', box_result['AP'], epoch)
        logger.add_scalar(log_group_name + '/AP50', box_result['AP50'], epoch)
        logger.add_scalar(log_group_name + '/AP75', box_result['AP75'], epoch)
        logger.add_scalar(log_group_name + '/APl', box_result['APl'], epoch)
        logger.add_scalar(log_group_name + '/APm', box_result['APm'], epoch)
        logger.add_scalar(log_group_name + '/APs', box_result['APs'], epoch)

    return preds

def train(args, epoch, loader, model, optimizer, device, logger=None):
    model.train()

    if get_rank() == 0:
        pbar = tqdm(enumerate(loader), total=len(loader), dynamic_ncols=True)
    else:
        pbar = enumerate(loader)

    for idx, (images, targets, _) in pbar:
        model.zero_grad()

        images = images.to(device)
        targets = [target.to(device) for target in targets]

        _, loss_dict = model(images, targets=targets)
        loss_cls = loss_dict['loss_cls'].mean()
        loss_box = loss_dict['loss_reg'].mean()
        loss_center = loss_dict['loss_centerness'].mean()

        loss = loss_cls + loss_box + loss_center
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()

        loss_reduced = reduce_loss_dict(loss_dict)
        loss_cls = loss_reduced['loss_cls'].mean().item()
        loss_box = loss_reduced['loss_reg'].mean().item()
        loss_center = loss_reduced['loss_centerness'].mean().item()

        if get_rank() == 0:
            pbar.set_description(
                (
                    f'epoch: {epoch + 1}; cls: {loss_cls:.4f}; '
                    f'box: {loss_box:.4f}; center: {loss_center:.4f}'
                )
            )

            # writing log to tensorboard
            if logger and idx % 10 == 0:
                totalStep = (epoch * len(loader) + idx) * args.batch * args.n_gpu
                logger.add_scalar('training/loss_cls', loss_cls, totalStep)
                logger.add_scalar('training/loss_box', loss_box, totalStep)
                logger.add_scalar('training/loss_center', loss_center, totalStep)
                logger.add_scalar('training/loss_all', (loss_cls + loss_box + loss_center), totalStep)


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return sampler.RandomSampler(dataset)

    else:
        return sampler.SequentialSampler(dataset)


if __name__ == '__main__':
    args = get_args()

    # Create working directory for saving intermediate results
    working_dir = args.working_dir
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)
    logger = SummaryWriter(working_dir)

    n_gpu = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    args.n_gpu = n_gpu
    args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='gloo', init_method='env://')
        synchronize()

    device = 'cuda'

    train_set = COCODataset(args.path, 'train', preset_transform(args, train=True))
    valid_set = COCODataset(args.path, 'val', preset_transform(args, train=False))

    # backbone = vovnet39(pretrained=True)
    # backbone = vovnet57(pretrained=True)
    # backbone = resnet18(pretrained=True)
    backbone = resnet50(pretrained=True)
    #backbone = resnet101(pretrained=True)
    model = ATSS(args, backbone)
    model = model.to(device)

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=0.0001,
        nesterov=True,
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.lr_steps, gamma=args.lr_gamma
    )

    if args.distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch,
        sampler=data_sampler(train_set, shuffle=True, distributed=args.distributed),
        num_workers=args.num_workers,
        collate_fn=collate_fn(args),
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=args.batch,
        sampler=data_sampler(valid_set, shuffle=False, distributed=args.distributed),
        num_workers=args.num_workers,
        collate_fn=collate_fn(args),
    )

    for epoch in range(args.epoch):
        train(args, epoch, train_loader, model, optimizer, device, logger=logger)
        valid(args, epoch, valid_loader, valid_set, model, device, logger=logger)

        scheduler.step()

        if get_rank() == 0:
            torch.save(
                {'model': model.module.state_dict(), 'optim': optimizer.state_dict()},
                working_dir + f'/epoch-{epoch + 1}.pt',
            )

