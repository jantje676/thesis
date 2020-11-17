import argparse
import torch
import numpy as np
from div_loss import cosine_loss, ssd, dpp, euclidean_heat_loss, euclidean_loss

"""
Script to calculate the different implemented div_loss scores on the layers datasetsd
"""
def main(args):
    versions = args.list_versions

    print(versions)
    for version in versions:
        scores = div_scores(args.data_folder, version)
        print("************** {} ***************".format(version))
        print("score cos: {:.4f}".format(scores[0]))
        print("score score_euc_heat: {:.4f}".format(scores[1]))
        print("score euc: {:.4f}".format(scores[2]))
        print("score ssd: {:.4f}".format(scores[3]))
        print("score dpp: {:.4f}".format(scores[4]))

def div_scores(data_folder, version):
    features = np.load("{}/data_ims_{}_train.npy".format(data_folder, version))
    length = features.shape[0]
    n_detectors = features.shape[1]
    features = torch.from_numpy(features)

    score_cos = 0
    score_euc_heat = 0
    score_euc = 0
    score_ssd = 0
    score_dpp = 0

    for i in range(length):
        print(i, end="\r", flush=True)
        feat = features[i].unsqueeze(dim=0)

        score_cos += cosine_loss(1, feat)
        score_euc_heat += euclidean_heat_loss(1, feat, 10000, n_detectors)
        score_euc += euclidean_loss(1, feat, n_detectors)
        score_ssd += ssd(1, feat)
        score_dpp += dpp(1,feat)

    score_cos = score_cos/length
    score_euc_heat = score_euc_heat/length
    score_euc = score_euc/length
    score_ssd = score_ssd/length
    score_dpp = score_dpp/length
    return (score_cos, score_euc_heat, score_euc, score_ssd, score_dpp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='visualize attention distribution')
    parser.add_argument('--data_folder', default="../data/Fashion200K/dresses", type=str, help='which features to use')
    parser.add_argument("--list_versions", nargs="+", default=["layers_alex", "layers_res", "layers_sim"])


    args = parser.parse_args()
    main(args)
