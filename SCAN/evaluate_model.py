from vocab import Vocabulary
import evaluation
import csv
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import glob
import argparse

def main(args):
    run = args.run
    checkpoint = args.checkpoint
    data_path = args.data_path
    version = args.version

    model_path = "{}{}/checkpoint/{}".format(args.model_path, run,checkpoint )
    plot_path = "{}{}/checkpoint/".format(args.plot_path,  run)
    rt, rti, attn = evaluation.evalrank(model_path, plot_path, run, version, data_path=args.data_path, split="test", vocab_path=args.vocab_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate trained SCAN model')

    parser.add_argument('--run', default="Run0", type=str, help='Which run to evaluate')
    parser.add_argument('--checkpoint', default="model_best.pth.tar", type=str, help='which checkpoint to use')
    parser.add_argument('--model_path', default="/$TMPDIR/runs/", type=str, help='which checkpoint to use')
    parser.add_argument('--data_path', default="/$TMPDIR/thesis/data/", type=str, help='which checkpoint to use')
    parser.add_argument('--vocab_path', default="/$TMPDIR/thesis/vocab/", type=str, help='which checkpoint to use')
    parser.add_argument('--plot_path', default="/$HOME/runs/", type=str, help='which checkpoint to use')
    parser.add_argument('--version', default="laenen", type=str, help='which version of features and vocab to use')

    args = parser.parse_args()
    main(args)
