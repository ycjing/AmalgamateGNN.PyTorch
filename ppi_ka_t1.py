import json
import numpy as np
import networkx as nx
from networkx.readwrite import json_graph
from dgl.data.ppi import PPIDataset


class Teacher1PPIDataset(PPIDataset):
    """Customized PPI Dataset for teacher GNN #1, inherited from dgl.data.ppi.PPIDataset

    Args:
        PPIDataset (dgl.data.ppi): dgl.data.ppi.PPIDataset 
    """

    def __init__(self, mode):
        """This is the function that initilizes the dataset

        Args:
            mode (string): ('train', 'valid', 'test')
        """

        assert mode in ['train', 'valid', 'test']
        self.mode = mode
        self._load()
        self._preprocess()

    def _preprocess(self):
        """This is the function that pre-processes the dataset
        """

        if self.mode == 'train':
            self.train_mask_list = []
            self.train_graphs = []
            self.train_labels = []
            for train_graph_id in range(1, 21):
                train_graph_mask = np.where(self.graph_id == train_graph_id)[0]
                self.train_mask_list.append(train_graph_mask)
                self.train_graphs.append(self.graph.subgraph(train_graph_mask))
                self.train_labels.append(self.labels[train_graph_mask][:,:61])
        if self.mode == 'valid':
            self.valid_mask_list = []
            self.valid_graphs = []
            self.valid_labels = []
            for valid_graph_id in range(21, 23):
                valid_graph_mask = np.where(self.graph_id == valid_graph_id)[0]
                self.valid_mask_list.append(valid_graph_mask)
                self.valid_graphs.append(self.graph.subgraph(valid_graph_mask))
                self.valid_labels.append(self.labels[valid_graph_mask][:,:61])
        if self.mode == 'test':
            self.test_mask_list = []
            self.test_graphs = []
            self.test_labels = []
            for test_graph_id in range(23, 25):
                test_graph_mask = np.where(self.graph_id == test_graph_id)[0]
                self.test_mask_list.append(test_graph_mask)
                self.test_graphs.append(self.graph.subgraph(test_graph_mask))
                self.test_labels.append(self.labels[test_graph_mask][:,:61])

    def __len__(self):
        """This is the function that returns the number of samples in this dataset

        Returns:
            integer: number of samples
        """
        
        if self.mode == 'train':
            return len(self.train_mask_list)
        if self.mode == 'valid':
            return len(self.valid_mask_list)
        if self.mode == 'test':
            return len(self.test_mask_list)

    def __getitem__(self, item):
        """This is the function that returns the i-th sample.

        Args:
            item (integer): the sample index

        Returns:
            tuple: a tuple containing the graphs, the node features as well as the corresponding labels
        """

        if self.mode == 'train':
            return self.train_graphs[item], self.features[self.train_mask_list[item]], self.train_labels[item]
        if self.mode == 'valid':
            return self.valid_graphs[item], self.features[self.valid_mask_list[item]], self.valid_labels[item]
        if self.mode == 'test':
            return self.test_graphs[item], self.features[self.test_mask_list[item]], self.test_labels[item]
            