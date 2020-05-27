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

# python evaluate_model.py --model_path "runs/" --run run8 --vocab_path "../vocab/" --version short --data_path "../data/"

def main(args):
    run = args.run
    checkpoint = args.checkpoint
    data_path = args.data_path
    version = args.version


    folders = glob.glob("{}{}/seed*/".format(args.model_path, run))
    print(folders)
    nr_runs = len(folders)


    r1 = 0
    r5 = 0
    r10 = 0
    medr = 0
    meanr = 0

    ri1 = 0
    ri5 = 0
    ri10 = 0
    medri = 0
    meanri = 0

    for i in range(nr_runs):
        print("Evaluating seed{}".format(str(i+1)))
        model_path = "{}{}/seed{}/checkpoint/{}".format(args.model_path, run, i+1, checkpoint )
        # plot_path = "{}{}/seed{}checkpoint/".format(args.plot_path,  run)
        rt, rti, attn, r, ri = evaluation.evalrank(model_path, run, version, data_path=args.data_path, split="test", vocab_path=args.vocab_path)
        r1 += r[0]
        r5 += r[1]
        r10 += r[2]
        medr += r[3]
        meanr += r[4]

        ri1 += ri[0]
        ri5 += ri[1]
        ri10 += ri[2]
        medri += ri[3]
        meanri += ri[4]

    r1 = r1 / nr_runs
    r5 = r5 / nr_runs
    r10 = r10 / nr_runs
    medr = medr / nr_runs
    meanr = meanr / nr_runs

    ri1 = ri1 / nr_runs
    ri5 = ri5 / nr_runs
    ri10 = ri10 / nr_runs
    medri = medri / nr_runs
    meanri = meanri / nr_runs

    print("AVERAGE Image to text: {:.1f} {:.1f} {:.1f} {:.1f} {:.1f}".format( r1, r5, r10, medr, meanr))
    print("AVERAGE Text to image: {:.1f} {:.1f} {:.1f} {:.1f} {:.1f}".format(ri1, ri5, ri10, medri, meanri))

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
