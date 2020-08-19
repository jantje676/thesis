import csv
import numpy as np
import shutil
import argparse

def main(opt):
    version = opt.version
    data_path = opt.data_path
    data_split = opt.data_split
    threshold = opt.threshold

    # read cptions
    captions = []
    caption_ids = []
    with open('{}/data_captions_{}_{}.txt'.format(data_path, version, data_split), 'r', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        for line in reader:
            captions.append(line[1].strip())
            caption_ids.append(line[0].strip())


    # read images
    images = np.load("{}/data_ims_{}_{}.npy".format(data_path, version, data_split))
    length = len(captions)

    # count frequency of words
    count = count_words(captions)

    # calculate normalized frequency count
    freq_score = calculatate_freq(captions, count)

    # upsample images
    captions_up, caption_ids_up, images_up = upsample(freq_score, captions, caption_ids, images)

    # concat captions and images
    captions_up = captions + captions_up
    caption_ids_up = caption_ids + caption_ids_up

    images_up = np.concatenate([images, images_up], axis = 0)

    # check if lengths match
    if len(captions_up) != images_up.shape[0]:
        print("lengths do not match")
        exit()

    # check how the frequencies have changes
    count_up = count_words(captions_up)

    # save images
    np.save( "{}/data_ims_{}_up_train.npy".format(data_path, version), images_up)

    # save captions
    with open('{}/data_captions_{}_up_train.txt'.format(data_path, version), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')

        for i in range(len(captions_up)):
            writer.writerow((caption_ids_up[i], captions_up[i]))

    # copy dev, test
    shutil.copy('{}/data_captions_{}_test.txt'.format(data_path, version),'{}/data_captions_{}_up_test.txt'.format(data_path, version))
    shutil.copy('{}/data_captions_{}_dev.txt'.format(data_path, version),'{}/data_captions_{}_up_dev.txt'.format(data_path, version))

    shutil.copy( "{}/data_ims_{}_test.npy".format(data_path, version), "{}/data_ims_{}_up_test.npy".format(data_path, version))
    shutil.copy( "{}/data_ims_{}_dev.npy".format(data_path, version), "{}/data_ims_{}_up_dev.npy".format(data_path, version))

def upsample(freq_score, captions, caption_ids, images):
    freq_score = np.asarray(freq_score)
    temp = np.where(freq_score < threshold)

    print("Added {} captions to upsampled dataset".format(len(temp[0])))

    captions_up = []
    caption_ids_up = []

    for i in range(len(temp[0])):
        captions_up.append(captions[temp[0][i]])
        caption_ids_up.append(caption_ids[temp[0][i]])

    # find images to be upsamnpled
    images_up = images[temp]

    return captions_up, caption_ids, images_up


# count the frequency of the words
def count_words(captions):
    count = {}
    for caption in captions:
        for word in caption.split(" "):
            if word in count.keys():
                count[word] += 1
            else:
                count[word] = 1
    minimum = min(count.values())
    maximum = max(count.values())
    return count

# calculate a frequency score of a caption
def calculatate_freq(captions, count):
    freq_score = []
    for caption in captions:
        caption = caption.split(" ")
        tot_freq = 0
        caption_l = len(caption)
        for word in caption:
            try:
                tot_freq += count[word]
            except:
                caption_length -= 1

        freq = tot_freq / caption_l
        freq_score.append(freq)

    freq_score = normalize(freq_score)

    return freq_score


def normalize(freq_score):
    max_freq = max(freq_score)
    min_freq = min(freq_score)
    for i in range(len(freq_score)):
        freq_score[i] = (freq_score[i] - min_freq)/(max_freq - min_freq)
    return freq_score

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--version', default='layers',
                        help='version')
    parser.add_argument('--data_path', default='../../data/Fashion200K',
                        help='path to datasets')
    parser.add_argument('--data_split', default='train',
                        help='split to choose')
    parser.add_argument('--threshold', default=0.25, type=float,
                        help='threshold to use frequent words')
    opt = parser.parse_args()
    main(opt)
