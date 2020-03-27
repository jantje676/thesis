from simclr import SimCLR
from data_aug.dataset_wrapper import DataSetWrapper
import argparse
from util import str2bool

def main(opt):
    dataset = DataSetWrapper(opt)
    simclr = SimCLR(dataset, opt)
    simclr.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=4, type=int, help='batch size')
    parser.add_argument('--epochs', default=80, type=int, help='epochs to run')
    parser.add_argument('--eval_every_n_epochs', default=1, type=int, help='when to evaluate')
    parser.add_argument('--fine_tune_from', default="None", type=str, help='load pretrained model')
    parser.add_argument('--log_every_n_steps', default=50, type=int, help='when to log')
    parser.add_argument('--weight_decay', default=10e-6, type=float, help='wheight decay')
    parser.add_argument('--fp16_precision', default=False, type=str2bool, help='fp16_precision')
    parser.add_argument('--output_dir', default="runs_simCLR/", type=str, help='directory to output runs')
    parser.add_argument('--config_file', default="./", type=str, help='./thesis/simCLR/ for lisa')
    parser.add_argument('--out_dim', default=256, type=int, help='output dimension')
    parser.add_argument('--base_model', default="resnet18", type=str, help='resnet18 or resnet50')
    parser.add_argument('--s', default=1, type=int, help='s for colorjitter')
    parser.add_argument('--input_shape_width', default=96, type=int, help='W, H, C')
    parser.add_argument('--input_shape_height', default=192, type=int, help='W, H, C')
    parser.add_argument('--num_workers', default=0 , type=int, help='num workers')
    parser.add_argument('--valid_size', default=0.1  , type=float, help='percentage for validation')
    parser.add_argument('--name_dataset', default="test_seg", type=str, help='which dataset')
    parser.add_argument('--dset_dir', default="../data", type=str, help='data diretory, ./thesis/data for lisa')
    parser.add_argument('--temperature', default=0.5, type=float, help='temperature for loss')
    parser.add_argument('--use_cosine_similarity', default=True, type=str2bool, help='use_cosine_similarity for loss')
    opt = parser.parse_args()
    main(opt)
