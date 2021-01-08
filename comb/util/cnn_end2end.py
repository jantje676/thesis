from collections import OrderedDict

import torchvision.models as models
import torch.nn as nn
import torch
import numpy as np
import torch.nn.init
from torchvision import transforms
import torch.utils.data as data
import csv
from utils import count_words, calculatate_freq, filter_freq, cut
import nltk
from PIL import Image
from util.segment_dresses import segment_dresses
import matplotlib.image as mpimg

# model to perform end2end training with a normal cnn
class CNN_end2end(nn.Module):
    def __init__(self, img_dim, embed_size):
        super(CNN_end2end, self).__init__()
        net = models.alexnet(pretrained=True)
        # take aways the last layers
        net.classifier = nn.Sequential(*[net.classifier[i] for i in range(5)])
        self.net = net
        self.fc = nn.Linear(img_dim, embed_size)

    def forward(self, x):
        # merge batch and segs and seperate again
        dim1 = x.shape[0]
        dim2 = x.shape[1]
        x = x.view(dim1 * dim2 , x.shape[2],x.shape[3],x.shape[4])
        out = self.net(x)
        out = out.view(dim1, dim2, -1)

        # bring to right embedding space
        out = self.fc(out)
        return out

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(CNN_end2end, self).load_state_dict(new_state)


class Data_segs(data.Dataset):
    def __init__(self, data_path, data_split, vocab, version, image_path, data_name):

        self.vocab = vocab
        loc = data_path + '/'
        self.captions = []
        self.images = []
        self.data_name = data_name
        self.image_path = image_path
        self.data_path = data_path


        with open('{}/data_captions_{}_{}.txt'.format(data_path, version, data_split), 'r', newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter='\t')
            for line in reader:
                self.captions.append(line[1].strip())
                self.images.append(line[0].strip())

        self.length = len(self.captions)
        self.im_div = 1
        self.count  = vocab.count
        freq_score, freqs = calculatate_freq(self.captions, self.count)
        self.freq_score = freq_score
        self.freqs = freqs

        self.transform = transforms.Compose([
            # transforms.Resize((256, 256)),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])])


    def __getitem__(self, index):
        # handle the image redundancy
        img_id = int(index/self.im_div)
        img_id = self.images[img_id]


        image = mpimg.imread("{}/{}/{}_0.jpeg".format(self.data_path, self.image_path, img_id))
        segments, bboxes = segment_dresses(image)

        image = stack_segments(segments, self.transform)

        caption = self.captions[index]
        vocab = self.vocab
        freq_score = self.freq_score[index]
        freqs = self.freqs[index]

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(
            str(caption).lower())

        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)

        return image, target, index, img_id, freq_score, freqs

    def __len__(self):
        return self.length


    def normal_tokenize(self,caption):
        # Convert caption (string) to word ids.
        tokens = self.tokenizer.word_tokenize(
            str(caption).lower())

        if self.filter:
            tokens = filter_freq(tokens, self.count, self.n_filter)

        if self.cut:
            tokens = cut(tokens, self.n_cut)
        caption = []
        caption.append(self.vocab('<start>'))
        caption.extend([self.vocab(token) for token in tokens])
        caption.append(self.vocab('<end>'))
        target = torch.Tensor(caption)
        return target

def stack_segments(segments, transform):
    segs = []
    for key in segments:
        seg_pil = Image.fromarray(segments[key])

        # transform images
        seg_transformed = transform(seg_pil)

        segs.append(seg_transformed)

    stacked_segments = torch.stack(segs, dim=0)
    return stacked_segments
