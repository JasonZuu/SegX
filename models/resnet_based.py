import torch
import torch.nn as nn
import torchvision.models as models


class ResNetBased(nn.Module):
    def __init__(self, num_classes=2, hidden_dims=256,
                 pretrained:bool=True):
        super().__init__()
        # Load a pre-trained ResNet50 as the embedding layers from torchvision.
        res_net = models.resnet50(pretrained=pretrained)
        in_features = res_net.fc.in_features  # ResNet's fully connected layer input size.

        # We keep all layers except the final fully connected layer (fc).
        self.embed_layers = nn.Sequential(*list(res_net.children())[:-2],
                                          nn.AdaptiveAvgPool2d((1, 1)))  # Global average pooling

        # Classifier for the task.
        self.classifier = nn.Sequential(nn.Linear(in_features, hidden_dims),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dims, num_classes))
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, use_sigmoid=True):
        y, _ = self.forward_with_hidden(x, use_sigmoid=use_sigmoid)
        return y
    
    def forward_with_hidden(self, x, use_sigmoid=True):
        z = self.embed_layers(x)
        z = z.view(z.size(0), -1)  # Flatten the output for the classifier.
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

    model = ResNetBased(num_classes=num_classes)
    model = model.to(device)

    y = model(images)
    print(y.size())  # torch.Size([2, 2])


if __name__ == "__main__":
    test_model()
