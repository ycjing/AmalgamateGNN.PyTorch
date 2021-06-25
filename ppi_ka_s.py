import json
import numpy as np
import networkx as nx
from networkx.readwrite import json_graph
from dgl.data.ppi import PPIDataset


class StudentPPIDataset(PPIDataset):
    """Customized PPI Dataset for the student GNN, inherited from dgl.data.ppi.PPIDataset

    Args:
        PPIDataset (dgl.data.ppi): dgl.data.ppi.PPIDataset 
    """

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
