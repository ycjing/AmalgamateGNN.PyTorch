import numpy as np
import torch as th
import torch.nn as nn
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax, GATConv
from torch.nn.parameter import Parameter


class KAGATConv(GATConv):
    """This is a customized GATConv inherited from dgl.nn.pytorch.conv.gatconv.GATConv,
       with the constructed unary edges, as described in Sect. 4.3 of the paper

    Args:
        GATConv (nn.Module): dgl.nn.pytorch.conv.gatconv.GATConv
    """

    def forward(self, graph, feat, grad=False):
        """This is the forward function of the customized GATConv

        Args:
            graph (DGLGraph): the input graphs containing the topological information
            feat (torch.Tensor): the input node features
            grad (bool, optional): whether to include the operation with the constructed unary edges 
                                   for the computation of the topological attributions. Defaults to False.

        Returns:
            torch.Tensor: the output activation of the KAGATConv layer
        """

        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    print('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, 'fc_src'):
                    self.fc_src, self.fc_dst = self.fc, self.fc
                feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            else:
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(
                    -1, self._num_heads, self._out_feats)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]

            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({'ft': feat_src, 'el': el})
            graph.dstdata.update({'er': er})

            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(graph.edata.pop('e'))

            # compute softmax
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
            
            # involve the operations with the constructed unary edges to obtain the topological attributions
            if grad and 'e_grad' in graph.edata:
                graph.edata['a'] = graph.edata['a'] * graph.edata['e_grad']

            # message passing
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']

            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
                rst = rst + resval

            # activation
            if self.activation:
                rst = self.activation(rst)

            return rst
            