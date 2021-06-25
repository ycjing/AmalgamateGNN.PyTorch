import torch
from utils import get_teacher1, get_teacher2, get_student


def collect_model(args, data_info_s, data_info_t1, data_info_t2):
    """This is the function that constructs the dictionary containing the models and the corresponding optimizers

    Args:
        args (parse_args): parser arguments
        data_info_s (dict): the dictionary containing the data information of the student
        data_info_t1 (dict): the dictionary containing the data information of teacher #1
        data_info_t2 (dict): the dictionary containing the data information of teacher #2

    Returns:
        dict: model dictionary ([model_name][model/optimizer])
    """

    device = torch.device("cpu") if args.gpu < 0 else torch.device("cuda:" + str(args.gpu))

    # initialize the two teacher GNNs and the student GNN
    s_model = get_student(args, data_info_s)        
    s_model.to(device)
    t1_model = get_teacher1(args, data_info_t1)                       
    t1_model.to(device)
    t2_model = get_teacher2(args, data_info_t2)      
    t2_model.to(device)

    # define the corresponding optimizers of the teacher GNNs and the student GNN
    params = s_model.parameters()
    s_model_optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    t1_model_optimizer = None
    t2_model_optimizer = None
   
    # construct the model dictionary containing the models and the corresponding optimizers
    model_dict = {}
    model_dict['s_model'] = {'model':s_model, 'optimizer':s_model_optimizer}
    model_dict['t1_model'] = {'model':t1_model, 'optimizer':t1_model_optimizer}
    model_dict['t2_model'] = {'model':t2_model, 'optimizer':t2_model_optimizer}

    return model_dict
    