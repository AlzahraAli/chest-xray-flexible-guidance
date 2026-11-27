
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import densenet121, DenseNet121_Weights


class ChestXRayClassifier(nn.Module):

    def __init__(self, pretrained=True):
        super().__init__()

        weights = (
            DenseNet121_Weights.IMAGENET1K_V1
            if pretrained else None
        )

        backbone = densenet121(weights=weights)

        self.features = backbone.features

        self.classifier = nn.Linear(
            backbone.classifier.in_features,
            1
        )

        self.temperature = 1.0

    def forward(self, images):
        features = F.relu(self.features(images))

        pooled = F.adaptive_avg_pool2d(
            features, (1, 1)
        ).flatten(1)

        logits = self.classifier(pooled).squeeze(1)

        weights = self.classifier.weight.squeeze(0)

        # Equation (1): weighted spatial scores
        scores = torch.einsum(
            "bkhw,k->bhw",
            features,
            weights
        )

        batch_size, height, width = scores.shape

        # Spatial softmax
        activation = F.softmax(
            scores.float().reshape(batch_size, -1)
            / self.temperature,
            dim=1
        )

        activation = activation.reshape(
            batch_size, height, width
        )

        return logits, activation


if __name__ == "__main__":
    from dataset import RSNADataset

    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    model = ChestXRayClassifier(pretrained=True)
    model = model.to(device)
    model.eval()

    dataset = RSNADataset("train")

    images = torch.stack([
        dataset[0][0],
        dataset[1][0]
    ]).to(device)

    with torch.no_grad():
        logits, activation = model(images)

        probabilities = torch.sigmoid(logits)

        activation_sums = activation.sum(dim=(1, 2))

    print("Input shape:", images.shape)
    print("Logits shape:", logits.shape)
    print("Activation shape:", activation.shape)
    print("Probabilities:", probabilities.cpu())
    print("Activation sums:", activation_sums.cpu())
    print("Finite logits:", torch.isfinite(logits).all().item())
    print("Finite activation:", torch.isfinite(activation).all().item())
    print("Nonnegative activation:", (activation >= 0).all().item())

    assert torch.allclose(
        activation_sums,
        torch.ones_like(activation_sums),
        atol=1e-6
    )

    print("Model test passed.")
