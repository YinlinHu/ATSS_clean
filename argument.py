import argparse


def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--epoch', type=int, default=12)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--n_save_sample', type=int, default=5)
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--working_dir', type=str, default="./training_dir/")
    parser.add_argument('--path', type=str, default="/data/COCO_17/")

    return parser

def get_args():
    parser = get_argparser()
    args = parser.parse_args()

    args.lr_steps = [8, 11]
    # args.lr_steps = [32, 44]
    args.lr_gamma = 0.1

    # args.feat_channels = [0, 0, 512, 768, 1024] # for vovnet
    # args.feat_channels = [0, 0, 128, 256, 512] # for resnet18
    args.feat_channels = [0, 0, 512, 1024, 2048] # for resnet50, resnet101
    args.out_channel = 256
    args.use_p5 = True
    # 
    args.n_class = 81
    args.n_conv = 4
    args.prior = 0.01
    # 
    args.inference_th = 0.05
    args.pre_nms_top_n = 1000
    args.nms_threshold = 0.6
    args.min_size = 0
    args.detections_per_img = 100
    # 
    # how to select positves: ATSS , SSC (FCOS), IoU (RetinaNet), TOPK
    args.positive_type = "ATSS"
    # regressing from a box ('BOX') or a point ('POINT')
    args.regression_type = "BOX"
    args.anchor_sizes = [64, 128, 256, 512, 1024]
    args.anchor_strides = [8, 16, 32, 64, 128]
    args.fg_iou_threshold = 0.5
    args.bg_iou_threshold= 0.4
    # topk for selecting candidate positive samples from each level
    args.top_k = 9
    # 
    args.reg_loss_weight = 2.0
    args.gamma = 2.0
    args.alpha = 0.25
    # 
    args.train_min_size_range = (640, 800)
    args.train_max_size = 1333
    args.test_min_size = 800
    args.test_max_size = 1333
    args.pixel_mean = [0.40789654, 0.44719302, 0.47026115]
    args.pixel_std = [0.28863828, 0.27408164, 0.27809835]
    args.size_divisible = 32

    return args
