import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--multigpu", type=bool, default=True)
    parser.add_argument("--device", type=str, default="0")

    # models
    parser.add_argument("--pretrained", type=bool, default=False)
    parser.add_argument("--input_size", type=int, default=800)

    # dataset
    parser.add_argument("--data_dir", type=str, default="./datasets/")
    parser.add_argument("--data_name", type=str, default='fruits')

    # training setting
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--lr_decay", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--n_epoch", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_epoch", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=1)

    # misc
    parser.add_argument("--ckpt_root", type=str, default="./FT_model")

    return parser.parse_args()


def make_template(opt):
    # dataset
    with open(os.path.join(opt.data_dir, opt.data_name, 'label.txt'), 'r') as f:
        lines = f.readlines()
    opt.num_classes = len(lines)

    # device
    opt.device_ids = [int(item) for item in opt.device.split(',')]
    if len(opt.device_ids) == 1:
        opt.multigpu = False
    opt.gpu = opt.device_ids[0]


def get_option():
    opt = parse_args()
    make_template(opt)
    return opt