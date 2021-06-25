import torch
import torch.nn as nn
import dgl.function as fn
import torch.nn.functional as F
from KAGATConv import KAGATConv


class GAT(nn.Module):
    """This is the definition of the GAT models

    Args:
        nn.Module: torch module
    """

    def __init__(self,
                 g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):

        super(GAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation

        # input projection (no residual)
        self.gat_layers.append(KAGATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, None))

        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(KAGATConv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, None))
                
        # output projection
        self.gat_layers.append(KAGATConv(
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None))

    def forward(self, inputs, middle=False):
        """This is the forward function of the GAT model

        Args:
            inputs (torch.Tensor): the input node features
            middle (bool, optional): whether to return the intermediate features 
                                     for visualizations. Defaults to False.

        Returns:
            torch.Tensor: the generated logits of the model
        """

        h = inputs
        middle_feats = []

        for l in range(self.num_layers):
            h = self.gat_layers[l](self.g, h).flatten(1)
            middle_feats.append(h)
            h = self.activation(h)
        
        # output projection
        logits = self.gat_layers[-1](self.g, h, grad=True).mean(1)

        if middle:
            return logits, middle_feats

        return logits
