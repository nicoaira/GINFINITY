from torch.utils.data import Dataset
from ginfinity.utils import dotbracket_to_graph, graph_to_tensor
import pandas as pd

class GINRNADataset(Dataset):
    def __init__(self, dataframe, graph_encoding = "standard"):
        #TODO: remove graph_encoding variable
        if isinstance(dataframe, str):  # Check if the input is a file path
            self.dataframe = pd.read_csv(dataframe, comment='#')
        else:
            self.dataframe = dataframe
        self.graph_encoding = graph_encoding

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        anchor_structure = self.dataframe.iloc[idx]["anchor_structure"]
        positive_structure = self.dataframe.iloc[idx]["positive_structure"]
        negative_structure = self.dataframe.iloc[idx]["negative_structure"]

        
        g_anchor = dotbracket_to_graph(anchor_structure)
        g_positive = dotbracket_to_graph(positive_structure)
        g_negative = dotbracket_to_graph(negative_structure)

        data_anchor = graph_to_tensor(g_anchor)
        data_positive = graph_to_tensor(g_positive)
        data_negative = graph_to_tensor(g_negative)
        
        
        return data_anchor, data_positive, data_negative
