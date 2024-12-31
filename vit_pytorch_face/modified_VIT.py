import torch
from torch import nn


class ModifiedViT(nn.Module):
    """
    Modified ViT model to return both classification output and embeddings

    Output shape: torch.Size([bs, 1000])
    Embeddings shape: torch.Size([bs, 768])
    """

    def __init__(self, vit_model):
        super(ModifiedViT, self).__init__()
        # Split the original ViT model and keep the submodules
        self.conv_proj = vit_model.conv_proj  # Convolutional projection
        self._process_input = vit_model._process_input  # Patch embedding
        self.class_token = vit_model.class_token
        self.encoder = vit_model.encoder  # Transformer encoder
        self.heads = vit_model.heads  # Classification head

    def forward(self, x, label):
        # label is not used in this model
        # Patch embedding
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to match the batch size
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)  # Transformer encoder
        embeddings = x[:, 0]  # [CLS] token's embedding

        output = self.heads(
            embeddings
        )  # The classification head calculates the final output

        return output, embeddings
