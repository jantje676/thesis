import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False),
                            "resnet50": models.resnet50(pretrained=False)}

        self.architecture = "global_pooling"

        resnet = self._get_basemodel(base_model)
        num_ftrs = resnet.fc.in_features

        if self.architecture == "normal":

            self.features = nn.Sequential(*list(resnet.children())[:-1])

            # projection MLP
            self.l1 = nn.Linear(num_ftrs, num_ftrs)
            self.l2 = nn.Linear(num_ftrs, out_dim)
        elif self.architecture == "global_pooling":
            self.features = nn.Sequential(*list(resnet.children())[:-2])
            # projection MLP

            self.features[7][0] = nn.Sequential(nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                                                nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                                nn.ReLU(inplace=True),
                                                nn.Conv2d(512, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                                                nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                                nn.ReLU(inplace=True))


            self.features[7][1] = nn.Sequential(nn.Conv2d(1024, 2048, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                                                nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                                nn.ReLU(inplace=True),
                                                nn.Conv2d(2048, 4096, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                                                nn.BatchNorm2d(4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                                nn.AdaptiveAvgPool2d((1,1)))


    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
            print("Feature extractor:", model_name)
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def forward(self, x):
        if self.architecture == "normal":
            h = self.features(x)
            h = h.squeeze()

            x = self.l1(h)
            x = F.relu(x)
            x = self.l2(x)
            return h, x
        elif self.architecture == "global_pooling":
            z = self.features(x)
            z = z.squeeze(dim=2)
            z = z.squeeze(dim=2)
            return z, z

    
