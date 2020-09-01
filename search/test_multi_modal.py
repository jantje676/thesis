import argparse
from evaluate_multi_modal import start_evaluation
from itertools import product
import os
import csv
import time
def main(args):
    combinations = get_params()

    if not os.path.exists(args.path_out):
        os.mkdir(args.path_out)

    with open('{}/results_eval_{}.txt'.format(args.path_out, time.time()), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(("alpha", "sum", "scan_sim", "r1", "r10", "r50"))

        for comb in combinations:
            args.sum = comb[0]
            args.scan_sim = comb[1]
            r1, r10, r50 = start_evaluation(args)

            writer.writerow((args.alpha, args.sum, args.scan_sim, r1, r10, r50))




def get_params():
    sum = ["max", "sum", "laenen"]
    scan = [True, False]
    combinations = list(product(sum, scan))
    return combinations
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multimodal search')
    parser.add_argument('--run', default="fscan3", type=str, help='name of run')
    parser.add_argument('--path_out', default="out_multi_eval", type=str, help='path to plots_scan')
    parser.add_argument('--vocab_path', default="../vocab", type=str, help='path to vocab')
    parser.add_argument('--data_path', default="../data", type=str, help='path to data')
    parser.add_argument('--search', default="multi", type=str, help='What kind of search to perform')
    parser.add_argument('--alpha', default=0.45, type=float, help='How much emphasis on i2i search for multimodal search')
    parser.add_argument('--similarity', default="sum", type=str, help='sum|max|laenen for t2i and i2i')
    parser.add_argument('--scan_sim', default=False, action='store_true', help='For t2i use scan similarity measure of cosine similarity')
    parser.add_argument('--model_folder', default="../comb", type=str, help='path to folder where models are stored')
    parser.add_argument('--nr_queries', default=3167, type=int, help='nr of queries to evaluate')
    args = parser.parse_args()
    main(args)
