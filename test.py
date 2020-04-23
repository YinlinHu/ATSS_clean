import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, sampler
from tqdm import tqdm

from argument import get_args
from backbone import vovnet39, resnet18, resnet50
from dataset import COCODataset, collate_fn
from model import ATSS
import transform
from evaluate import evaluate
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    DistributedSampler,
    all_gather,
)
from train import (
    accumulate_predictions,
    valid,
    data_sampler,
)
from coco_meta import CLASS_NAME
from visualize import show_bbox

def save_predictions_to_images(dataset, predictions):
    # 
    if get_rank() != 0:
        return
        
    for id, pred in enumerate(predictions):
        orig_id = dataset.id2img[id]

        if len(pred) == 0:
            continue

        img_meta = dataset.get_image_meta(id)
        width = img_meta['width']
        height = img_meta['height']
        pred = pred.resize((width, height))
        
        boxes = pred.bbox.tolist()
        scores = pred.get_field('scores').tolist()
        ids = pred.get_field('labels').tolist()

        img_name = img_meta['file_name']
        img_baseName = os.path.splitext(img_name)[0]
        # 
        print('saving ' + img_name + ' ...')
        imgroot = dataset.root
        show_bbox(imgroot + '/' + img_name, boxes, ids, CLASS_NAME, file_name=img_name, scores=scores)

        categories = [dataset.id2category[i] for i in ids]
        for k, box in enumerate(boxes):
            category_id = categories[k]
            score = scores[k]
            # csv_file.write("%s,%s.xml,%f,%d,%d,%d,%d\n" % (CLASS_NAME[category_id], img_baseName, score,\
            #     int(box[0] + 0.5), int(box[1] + 0.5), int(box[2] + 0.5), int(box[3] + 0.5)))

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'

    args = get_args()

    n_gpu = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='gloo', init_method='env://')
        synchronize()

    device = 'cuda'

    valid_trans = transform.Compose(
        [
            transform.Resize(args.test_min_size, args.test_max_size),
            transform.ToTensor(), 
            transform.Normalize(args.pixel_mean, args.pixel_std)
        ]
    )

    valid_set = COCODataset("/data/COCO_17/", 'val', valid_trans)

    # backbone = vovnet39(pretrained=False)
    # backbone = resnet18(pretrained=False)
    backbone = resnet50(pretrained=False)
    model = ATSS(args, backbone)

    # load weight
    model_file = "./training_dir/epoch-12.pt"
    chkpt = torch.load(model_file, map_location='cpu')  # load checkpoint
    model.load_state_dict(chkpt['model'])
    print('load weights from ' + model_file)

    model = model.to(device)
    
    if args.distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

    valid_loader = DataLoader(
        valid_set,
        batch_size=args.batch,
        sampler=data_sampler(valid_set, shuffle=False, distributed=args.distributed),
        num_workers=args.num_workers,
        collate_fn=collate_fn(args),
    )

    predictions = valid(args, 0, valid_loader, valid_set, model, device)
    save_predictions_to_images(valid_set, predictions)


