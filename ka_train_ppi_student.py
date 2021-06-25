import os
import numpy as np
import torch
import torch.nn.functional as F
from os.path import join
import dgl
import argparse
import time
import random
from ka_gat import GAT
from utils import get_data_loader, load_checkpoint, val_model, test_model, generate_label
from ka_loss import optimizing, loss_fn_kd, gen_mi_attrib_loss
from ka_model import collect_model
from topological_attrib_t import ATTNET_t
from topological_attrib_s import ATTNET_s

torch.set_num_threads(1)


def train_student(args, auxiliary_model, data, device):
    """This is the function that trains the student GNN by knowledge amalgamation 
 
    Args:
        args (parse_args): parser arguments
        auxiliary_model (dict): model dictionary ([model_name][model/optimizer])
        data (list): dataloader for training, validating and testing
        device (torch.device): device to place the pytorch tensor
    """

    # record the best validation F1 score and loss value
    best_score = 0  
    best_loss = 1000.0

    # dataloader for training, validating, and testing
    train_dataloader, valid_dataloader, test_dataloader = data
    
    # multi-label loss function
    loss_fcn = torch.nn.BCEWithLogitsLoss()

    # get the two pre-trained teacher GNNs and an initialized student GNN
    t1_model = auxiliary_model['t1_model']['model']
    t2_model = auxiliary_model['t2_model']['model']
    s_model = auxiliary_model['s_model']['model']

    # get the topological attribution nets for teachers and student
    attrib_net_t1 = ATTNET_t(t1_model, args)
    attrib_net_t2 = ATTNET_t(t2_model, args)
    attrib_net_s = ATTNET_s(s_model, args)

    # training for epochs
    for epoch in range(1, args.s_epochs + 1):
        s_model.train()

        # initializing the lists to record the loss
        loss_list = [] # total loss
        soft_loss_list = [] # soft target loss
        attrib_loss_list = [] # topological semantic alignment loss 

        t0 = time.time()
        
        for _, subgraph in enumerate(train_dataloader):
            subgraph = subgraph.to(device)
            feats = subgraph.ndata['feat'].float()

            s_model.g = subgraph
            for layer in s_model.gat_layers:
                layer.g = subgraph
            
            # inference results of the trainable student GNN
            logits, _ = s_model(feats.float(), middle=True)
            
            # generate soft labels from pre-trained teacher GNNs (no_grad)
            logits_t1 = generate_label(t1_model, subgraph, feats, device)
            logits_t2 = generate_label(t2_model, subgraph, feats, device)

            # compute soft target loss by using soft labels
            class_loss_t1, labels_t1 = loss_fn_kd(logits[:,:61], logits_t1)
            class_loss_t2, labels_t2 = loss_fn_kd(logits[:,61:], logits_t2)
            soft_loss = class_loss_t1 + class_loss_t2

            # compute topological attribution maps for two teachers and the student
            attrib_map_t1 = attrib_net_t1.observe(subgraph, feats, labels_t1)
            attrib_map_t2 = attrib_net_t2.observe(subgraph, feats, labels_t2)
            attrib_map_st1 = attrib_net_s.observe(subgraph, feats, labels_t1, mode='t1')
            attrib_map_st2 = attrib_net_s.observe(subgraph, feats, labels_t2, mode='t2')

            # compute topological semantics alignment loss
            mi_attrib_loss = gen_mi_attrib_loss(subgraph, attrib_map_t1, attrib_map_t2, attrib_map_st1, attrib_map_st2)
            
            # total loss (Eq. 3 in the paper)
            loss = soft_loss + args.attrib_weight * mi_attrib_loss

            # model optimization
            optimizing(auxiliary_model, loss, ['s_model'])

            loss_list.append(loss.item())
            soft_loss_list.append(soft_loss.item())
            attrib_loss_list.append(mi_attrib_loss.item())

        loss_data = np.array(loss_list).mean()
        soft_loss_data = np.array(soft_loss_list).mean()
        attrib_loss_data = np.array(attrib_loss_list).mean()

        # print loss (Grad Loss here denotes the topological semantics alignment loss)
        print(f"Epoch {epoch:05d} | Loss: {loss_data:.4f} | Grad Loss: {attrib_loss_data:.4f} | Soft Loss: {soft_loss_data:.4f} | Time: {time.time()-t0:.4f}s") 

        if (epoch + 1) % 5 == 0:
           score = val_model(valid_dataloader, s_model, device, loss_fcn)
           # we report the results with the best validation F1 score or the smallest loss data
           if score > best_score or loss_data < best_loss:
               best_score = score
               best_loss = loss_data
               test_model(test_dataloader, s_model, device, loss_fcn)

    print('=====================') # end training


def main(args):
    device = torch.device("cpu") if args.gpu < 0 else torch.device("cuda:" + str(args.gpu))

    # get the dataloader and the associated data information
    _, data_info_t1 = get_data_loader(args, 'teacher1', device)
    _, data_info_t2 = get_data_loader(args, 'teacher2', device)
    data_s, data_info_s = get_data_loader(args, 'student', device)

    # model dictionary that contains all the models
    model_dict = collect_model(args, data_info_s, data_info_t1, data_info_t2)

    # load the two pre-trained teacher GNNs
    t1_model = model_dict['t1_model']['model']
    load_checkpoint(t1_model, "./teacher_models/t1_best_model.pt", device)
    t2_model = model_dict['t2_model']['model']
    load_checkpoint(t2_model, "./teacher_models/t2_best_model.pt", device)

    # train a multi-talented student GNN
    train_student(args, model_dict, data_s, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AmalgamateGNN')
    parser.add_argument("--gpu", type=int, default=0,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--residual", action="store_true", default=True,
                        help="use residual connection")
    parser.add_argument("--in-drop", type=float, default=0,
                        help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=0,
                        help="attention dropout")
    parser.add_argument('--alpha', type=float, default=0.2,
                        help="the negative slop of leaky relu")
    parser.add_argument('--batch-size', type=int, default=2,
                        help="batch size used for training, validation and test")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=4e-6,
                        help="weight decay")
    parser.add_argument("--t1-num-heads", type=int, default=4,
                        help="number of hidden attention heads of teacher #1")
    parser.add_argument("--t2-num-heads", type=int, default=4,
                        help="number of hidden attention heads of teacher #2")
    parser.add_argument("--t1-num-out-heads", type=int, default=6,
                        help="number of output attention heads of teacher #1")
    parser.add_argument("--t2-num-out-heads", type=int, default=6,
                        help="number of output attention heads of teacher #2")
    parser.add_argument("--t1-num-layers", type=int, default=2,
                        help="number of hidden layers of teacher #1")
    parser.add_argument("--t2-num-layers", type=int, default=2,
                        help="number of hidden layers of teacher #2")
    parser.add_argument("--t1-num-hidden", type=int, default=256,
                        help="number of hidden units of teacher #1")
    parser.add_argument("--t2-num-hidden", type=int, default=256,
                        help="number of hidden units of teacher #2")
    parser.add_argument("--s-epochs", type=int, default=1500,
                        help="number of training epochs")
    parser.add_argument("--s-num-heads", type=int, default=4,
                        help="number of hidden attention heads of the student")
    parser.add_argument("--s-num-out-heads", type=int, default=6,
                        help="number of output attention heads of the student")
    parser.add_argument("--s-num-layers", type=int, default=2,
                        help="number of hidden layers of the student")
    parser.add_argument("--s-num-hidden", type=int, default=256,
                        help="number of hidden units of the student")
    parser.add_argument('--attrib-weight', type=float, default=0.10,
                        help="weight coeff of the topological semantics alignment loss")
    args = parser.parse_args()
    print(args)

    seed = 1234
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    main(args)
