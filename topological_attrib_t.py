import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
import torch.autograd as autograd


def get_attrib(net, graph, features, labels):
    """This is the function that computes the topological attributions for the teacher

    Args:
        net (nn.Module): the teacher GNN model
        graph (DGLGraph): the input graphs containing the topological information
        features (torch.Tensor): the input node features
        labels (torch.Tensor): the soft labels

    Returns:
        torch.Tensor: topological attributions
    """

    labels = torch.where(labels > 0.0, 
                            torch.ones(labels.shape).to(labels.device), 
                            torch.zeros(labels.shape).to(labels.device)).type(torch.bool)

    # zero gradients
    if net.g.edata['e_grad'].grad is not None:
        net.g.edata['e_grad'].grad.zero_()

    # generate model outputs
    output = net(features.float())
    
    # set the gradients of the corresponding output activations to one 
    output_grad = torch.zeros_like(output)
    output_grad[labels] = 1

    # compute the gradients
    attrib = autograd.grad(outputs=output, inputs=net.g.edata['e_grad'], grad_outputs=output_grad, only_inputs=True)[0]

    return attrib
    

class ATTNET_t(nn.Module):
    """This is the class that returns the topological attribution maps of the teacher GNN

    Args:
        nn.Module: torch module
    """
    
    def __init__(self,
                 model,
                 args):

        super(ATTNET_t, self).__init__()
        # set up the network
        self.net = model

    def forward(self, graph, features):
        """This is the forward function of ATTNET_t

        Args:
            graph (DGLGraph): the input graphs containing the topological information
            features (torch.Tensor): the input node features

        Returns:
            torch.Tensor: the generated logits of the model
        """

        self.net.g = graph
        for layer in self.net.gat_layers:
            layer.g = graph
        output = self.net(features)

        return output

    def observe(self, graph, features, labels):
        """This is the function that returns the topological attribution maps

        Args:
            graph (DGLGraph): the input graphs containing the topological information
            features (torch.Tensor): the input node features
            labels (torch.Tensor): the soft labels

        Returns:
            torch.Tensor: topological attributions
        """

        self.net.train()
        self.net.g = graph
        for layer in self.net.gat_layers:
            layer.g = graph
            
        # set auxiliary unary features for the edges to obtain the topological attributions
        self.net.g.edata['e_grad'] = torch.cuda.FloatTensor( [1.0] 
                                * self.net.g.number_of_edges() ).view((-1, 1, 1))
        self.net.g.edata['e_grad'].requires_grad = True
    
        attrib = get_attrib(self.net, graph, features, labels)

        self.net.eval()
        
        return attrib
