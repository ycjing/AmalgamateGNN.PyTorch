import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import f1_score
import dgl
from dgl.data.ppi import PPIDataset
from ka_gat import GAT
from ppi_ka_s import StudentPPIDataset
from ppi_ka_t1 import Teacher1PPIDataset
from ppi_ka_t2 import Teacher2PPIDataset


def evaluate(feats, model, graph, labels, loss_fcn):
    """This is the function that computes the F1 score of the student model

    Args:
        feats (torch.Tensor): the input node features
        model (nn.Module): the student GNN model
        graph (DGLGraph): the input graphs containing the topological information
        labels (torch.Tensor): the soft labels
        loss_fcn (torch.nn): multi-label loss function

    Returns:
        tuple: a tuple containing the F1 scores for the two tasks of the teachers as well as the loss data
    """

    model.eval()

    with torch.no_grad():
        model.g = graph
        for layer in model.gat_layers:
            layer.g = graph
        output = model(feats.float())
        loss_data = loss_fcn(output, labels.float())
        predict = np.where(output.data.cpu().numpy() > 0.0, 1, 0)
        score_whole = f1_score(labels.data.cpu().numpy(),
                         predict, average='micro')
        
        # F1 score for the task of teacher #1
        score_part1 = f1_score(labels.data.cpu().numpy()[:,:61],
                         predict[:,:61], average='micro')

        # F1 score for the task of teacher #2
        score_part2 = f1_score(labels.data.cpu().numpy()[:,61:],
                         predict[:,61:], average='micro')

    model.train()
    
    return score_whole, score_part1, score_part2, loss_data.item()


def test_model(test_dataloader, model, device, loss_fcn):
    """This is the function that returns the testing F1 scores of the student GNN

    Args:
        test_dataloader (torch.utils.data.DataLoader): testing dataloader for the student GNN
        model (nn.Module): the student GNN model
        device (torch.device): device to place the pytorch tensor
        loss_fcn (torch.nn): multi-label loss function
    """

    test_score_list = []
    test_score_part1_list = []
    test_score_part2_list = []

    model.eval()

    with torch.no_grad():
        for _, graph in enumerate(test_dataloader):
            graph = graph.to(device)
            feats = graph.ndata['feat'].float()
            labels = graph.ndata['label'].float()

            test_score_list.append(evaluate(feats, model, graph, labels.float(), loss_fcn)[0])
            test_score_part1_list.append(evaluate(feats, model, graph, labels.float(), loss_fcn)[1])
            test_score_part2_list.append(evaluate(feats, model, graph, labels.float(), loss_fcn)[2])

        mean_score = np.array(test_score_list).mean()
        mean_score_part1 = np.array(test_score_part1_list).mean()
        mean_score_part2 = np.array(test_score_part2_list).mean()

        print(f"F1-Score on testset:       whole {mean_score:.4f}, part1 {mean_score_part1:.4f}, part2 {mean_score_part2:.4f}")
    
    model.train()

    return


def generate_label(t_model, graph, feats, device):
    """This is the function that generates the logits for the unlabeled data from the pre-trained teacher GNNs

    Args:
        t_model (nn.Module): the pre-trained teacher GNN models
        graph (DGLGraph): the input graphs containing the topological information
        feats (torch.Tensor): the input node features
        device (torch.device): device to place the pytorch tensor

    Returns:
        torch.Tensor: logits of the teacher GNNs
    """
    
    t_model.eval()

    with torch.no_grad():
        t_model.g = graph
        for layer in t_model.gat_layers:
            layer.g = graph
    
        # generate logits from the pre-trained teacher models
        logits_t = t_model(feats.float())

    return logits_t.detach()
    

def val_model(valid_dataloader, model, device, loss_fcn):
    """This is the function that returns the validation F1 scores of the student GNN

    Args:
        valid_dataloader (torch.utils.data.DataLoader): validation dataloader for the student GNN
        model (nn.Module): the student GNN model
        device (torch.device): device to place the pytorch tensor
        loss_fcn (torch.nn): multi-label loss function

    Returns:
        torch.Tensor: the overall F1 score on the validation sets
    """

    val_score_list = []
    val_score_part1_list = []
    val_score_part2_list = []

    model.eval()

    with torch.no_grad():
        for _, graph in enumerate(valid_dataloader):
            graph = graph.to(device)
            feats = graph.ndata['feat'].float()
            labels = graph.ndata['label'].float()

            val_score_list.append(evaluate(feats, model, graph, labels.float(), loss_fcn)[0])
            val_score_part1_list.append(evaluate(feats, model, graph, labels.float(), loss_fcn)[1])
            val_score_part2_list.append(evaluate(feats, model, graph, labels.float(), loss_fcn)[2])

        mean_score = np.array(val_score_list).mean()
        mean_score_part1 = np.array(val_score_part1_list).mean()
        mean_score_part2 = np.array(val_score_part2_list).mean()

        print(f"F1-Score on valset:       whole {mean_score:.4f}, part1 {mean_score_part1:.4f}, part2 {mean_score_part2:.4f}")

    model.train()

    return mean_score   


def get_teacher1(args, data_info):
    """This is the function that returns the model architecture of teacher #1

    Args:
        args (parse_args): parser arguments
        data_info (dict): the dictionary containing the data information of teacher #1

    Returns:
        model: teacher model #1
    """

    heads = ([args.t1_num_heads] * args.t1_num_layers) + [args.t1_num_out_heads]

    model = GAT(data_info['g'],
            args.t1_num_layers,
            data_info['num_feats'],
            args.t1_num_hidden,
            data_info['n_classes'],
            heads,
            F.elu,
            args.in_drop,
            args.attn_drop,
            args.alpha,
            args.residual)

    return model


def get_teacher2(args, data_info):
    """This is the function that returns the model architecture of teacher #2

    Args:
        args (parse_args): parser arguments
        data_info (dict): the dictionary containing the data information of teacher #2

    Returns:
        model: teacher model #2
    """

    heads = ([args.t2_num_heads] * args.t2_num_layers) + [args.t2_num_out_heads]

    model = GAT(data_info['g'],
            args.t2_num_layers,
            data_info['num_feats'],
            args.t2_num_hidden,
            data_info['n_classes'],
            heads,
            F.elu,
            args.in_drop,
            args.attn_drop,
            args.alpha,
            args.residual)

    return model


def get_student(args, data_info):
    """This is the function that returns the model architecture of the student

    Args:
        args (parse_args): parser arguments
        data_info (dict): the dictionary containing the data information of the student

    Returns:
        model: the student model
    """

    heads = ([args.s_num_heads] * args.s_num_layers) + [args.s_num_out_heads]

    model = GAT(data_info['g'],
            args.s_num_layers,
            data_info['num_feats'],
            args.s_num_hidden,
            data_info['n_classes'],
            heads,
            F.elu,
            args.in_drop,
            args.attn_drop,
            args.alpha,
            args.residual)

    return model


def collate(graphs):
    """This is the function that collates lists of samples into batches

    Args:
        graphs (DGLGraph): the input graphs containing the topological information

    Returns:
        DGLGraph: batched graphs
    """
    
    graph = dgl.batch(graphs)
    return graph


def get_data_loader(args, type, device):
    """This is the function that returns the dataloaders and the data information

    Args:
        args (parse_args): parser arguments
        type (string): ('teacher1', 'teacher2', 'student')
        device (torch.device): device to place the dataloader

    Returns:
        list: the dataloaders and the associated data information
    """

    # obtain the dataloaders
    train_dataset = PPIDataset(mode='train')
    valid_dataset = PPIDataset(mode='valid')
    test_dataset = PPIDataset(mode='test')  

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate, num_workers=4, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=collate, num_workers=2)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate, num_workers=2)

    # get the data information
    data_info = {}

    g = train_dataset[0]
    n_classes = train_dataset.num_labels
    num_feats = g.ndata['feat'].shape[1]
    data_info['num_feats'] = num_feats
    data_info['g'] = g.int().to(device)

    if type == 'teacher1':
        data_info['n_classes'] = 61
    elif type == 'teacher2':
        data_info['n_classes'] = 60
    elif type == 'student':
        assert(n_classes == 121)
        data_info['n_classes'] = n_classes    

    return (train_dataloader, valid_dataloader, test_dataloader), data_info


def save_checkpoint(model, path):
    """This is the function that saves the checkpoint

    Args:
        model (nn.Module): the trained GNN model
        path (string): directory to save the model
    """

    dirname = os.path.dirname(path)

    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    
    torch.save(model.state_dict(), path)
    print(f"save model to {path}")


def load_checkpoint(model, path, device):
    """This is the function that loads the checkpoint

    Args:
        model (nn.Module): the GNN model
        path (string): directory to load the model
        device (torch.device): device to place the model
    """

    model.load_state_dict(torch.load(path, map_location=device))
    print(f"Load model from {path}")
