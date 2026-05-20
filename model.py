# model.py
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class WildfireDetector(nn.Module):
    """
    EfficientNet-B0 fine-tuned for binary wildfire classification.

    Transfer learning strategy:
      - Keep all pretrained convolutional layers FROZEN initially
      - Only train the new classifier head we attach
      - After a few epochs, unfreeze the whole network ("fine-tuning")
    """

    def __init__(self, num_classes: int = 1, freeze_backbone: bool = True):
        super().__init__()

        # Load EfficientNet-B0 with ImageNet pretrained weights
        self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

        # How many features does the backbone output?
        # EfficientNet-B0 outputs 1280 features before the classifier
        in_features = self.backbone.classifier[1].in_features

        # Replace the original ImageNet classifier (1000 classes)
        # with our own binary head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),  # dropout for regularization
            nn.Linear(in_features, num_classes)  # single output = fire probability
        )

        if freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self):
        """
        Freeze all convolutional layers.
        Only our new classifier head will be trained initially.
        This is crucial for beginners: it makes training fast and stable.
        """
        for param in self.backbone.features.parameters():
            param.requires_grad = False
        print("Backbone frozen. Only classifier head will train.")

    def unfreeze_backbone(self):
        """
        Call this after initial training to fine-tune the whole network.
        Usually after 5–10 epochs with frozen backbone.
        """
        for param in self.backbone.features.parameters():
            param.requires_grad = True
        print("Backbone unfrozen. Full model will fine-tune.")

    def forward(self, x):
        # x shape: [batch_size, 3, 224, 224]
        # Returns raw logits (not probabilities!)
        # Use BCEWithLogitsLoss which applies sigmoid internally
        return self.backbone(x).squeeze(1)  # shape: [batch_size]

    @torch.no_grad()
    def predict_with_uncertainty(self, x: "torch.Tensor", n_samples: int = 30):
        """
        Monte Carlo Dropout inference.

        Runs n_samples stochastic forward passes with Dropout kept active
        (model stays in train() mode so dropout fires each pass), then
        returns the mean and standard deviation of the predicted fire
        probabilities.

        The head has Dropout(p=0.3), so each pass samples a different
        sub-network — this gives an empirical posterior over predictions.

        Returns:
            mean_prob  — shape [batch], mean fire probability
            std_prob   — shape [batch], epistemic uncertainty (1σ)

        Interpretation:
            std < 0.05  → model is confident
            std 0.05–0.12 → moderate uncertainty
            std > 0.12  → model is unsure; treat prediction cautiously
        """
        was_training = self.training
        self.train()   # enable dropout for stochastic sampling
        probs = torch.stack([
            torch.sigmoid(self.forward(x)) for _ in range(n_samples)
        ])                  # [n_samples, batch]
        if not was_training:
            self.eval()
        return probs.mean(0), probs.std(0)