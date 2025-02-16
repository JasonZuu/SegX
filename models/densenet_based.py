import torch
import torch.nn as nn
import torchvision.models as models


class DenseNetBased(nn.Module):
    def __init__(self, num_classes=2, hidden_dims=256,
                 pretrained:bool=True):
        super().__init__()
        # Load a pre-trained DenseNet121 as the embedding layers from torchvision.
        dense_net = models.densenet121(pretrained=pretrained)
        in_features = dense_net.classifier.in_features

        # The features before the classifier in DenseNet121 are accessed differently.
        self.embed_layers = nn.Sequential(*list(dense_net.features),
                                          nn.ReLU(inplace=True),
                                          nn.AdaptiveAvgPool2d((1,1)))
        
        self.classifier = nn.Sequential(nn.Linear(in_features, hidden_dims),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dims, num_classes))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, use_sigmoid=True):
        y, _ = self.forward_with_hidden(x, use_sigmoid=use_sigmoid)
        return y
    

    def forward_with_hidden(self, x, use_sigmoid=True):
        z = self.embed_layers(x)
        z = z.view(z.size(0), -1) # Flatten the output for the classifier.
        y = self.classifier(z)
        if use_sigmoid:
            y = self.sigmoid(y)
        hidden = {"embed": z}
        return y, hidden


def test_model():
    num_classes = 2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    images = torch.randn(2, 3, 512, 512)
    images = images.to(device)

    model = DenseNetBased(num_classes=num_classes)
    model = model.to(device)

    y = model(images)
    print(y.size())  # torch.Size([2, 2])


if __name__ == "__main__":
    test_model()
