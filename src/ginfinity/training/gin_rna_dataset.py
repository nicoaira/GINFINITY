from torch.utils.data import Dataset
from ginfinity.utils import dotbracket_to_graph, graph_to_tensor
import pandas as pd
import torch


class GINRNADataset(Dataset):
    def __init__(self, dataframe, graph_encoding="standard", seq_weight: float = 0.0):
        """Triplet dataset for RNA structures.

        Parameters
        ----------
        dataframe : Union[pd.DataFrame, str]
            DataFrame or path to CSV/TSV containing the structures and sequences.
        graph_encoding : str, optional
            Kept for backwards compatibility; currently unused.
        seq_weight : float, optional
            Weight for nucleotide one-hot features relative to pairing
            information. ``0`` disables sequence features.
        """
        if isinstance(dataframe, str):  # Check if the input is a file path
            self.dataframe = pd.read_csv(dataframe, comment='#')
        else:
            self.dataframe = dataframe
        self.graph_encoding = graph_encoding
        self.seq_weight = seq_weight

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        anchor_structure = row["anchor_structure"]
        positive_structure = row["positive_structure"]
        negative_structure = row["negative_structure"]

        anchor_seq = row.get("anchor_seq")
        positive_seq = row.get("positive_seq")
        negative_seq = row.get("negative_seq")

        g_anchor = dotbracket_to_graph(anchor_structure, anchor_seq)
        g_positive = dotbracket_to_graph(positive_structure, positive_seq)
        g_negative = dotbracket_to_graph(negative_structure, negative_seq)

        data_anchor = graph_to_tensor(g_anchor, self.seq_weight)
        data_positive = graph_to_tensor(g_positive, self.seq_weight)
        data_negative = graph_to_tensor(g_negative, self.seq_weight)

        return data_anchor, data_positive, data_negative


class GINRNAPairDataset(Dataset):
    """Dataset yielding pairs of structures and a scalar target."""

    def __init__(self, dataframe, graph_encoding="standard", seq_weight: float = 0.0):
        if isinstance(dataframe, str):
            self.dataframe = pd.read_csv(dataframe, comment="#")
        else:
            self.dataframe = dataframe
        self.graph_encoding = graph_encoding
        self.seq_weight = seq_weight

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        anchor_structure = row["anchor_structure"]
        positive_structure = row["positive_structure"]
        target = row["f_total_modifications"]

        anchor_seq = row.get("anchor_seq")
        positive_seq = row.get("positive_seq")

        g_anchor = dotbracket_to_graph(anchor_structure, anchor_seq)
        g_positive = dotbracket_to_graph(positive_structure, positive_seq)

        data_anchor = graph_to_tensor(g_anchor, self.seq_weight)
        data_positive = graph_to_tensor(g_positive, self.seq_weight)

        target = torch.tensor([float(target)], dtype=torch.float32)
        return data_anchor, data_positive, target
