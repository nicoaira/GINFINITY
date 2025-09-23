from torch.utils.data import Dataset
from ginfinity.utils import dotbracket_to_graph, graph_to_tensor
import pandas as pd
import torch
from typing import Dict, List, Any


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


class GINAlignmentDataset(Dataset):
    """Dataset providing groups of structures that belong to the same alignment."""

    def __init__(
        self,
        dataframe,
        alignment_map: Dict[str, Dict[str, Dict[str, int]]],
        graph_encoding: str = "standard",
        seq_weight: float = 0.0,
        structure_column: str = "structure",
    ):
        if isinstance(dataframe, str):
            df = pd.read_csv(dataframe, comment="#")
        else:
            df = dataframe.copy()

        self.graph_encoding = graph_encoding
        self.seq_weight = seq_weight
        self.alignment_groups: List[Dict[str, Any]] = []
        self.alignment_map = alignment_map

        for alignment_id, group_df in df.groupby("alignment_id"):
            structures: List = []
            for _, row in group_df.iterrows():
                structure = row[structure_column]
                sequence = row.get("sequence")
                g = dotbracket_to_graph(structure, sequence)
                data = graph_to_tensor(g, self.seq_weight)

                sequence_id = row.get("sequence_id")
                if pd.notna(sequence_id):
                    try:
                        sequence_id_int = int(sequence_id)
                    except (TypeError, ValueError):
                        sequence_id_int = sequence_id
                else:
                    sequence_id_int = None

                data.sequence_id = sequence_id_int
                data.alignment_id = alignment_id
                data.binary_code = row.get("binary_code")

                mapping = self._resolve_alignment_mapping(alignment_id, sequence_id_int)
                data.alignment_mapping = {str(k): v for k, v in mapping.items()}
                data.alignment_positions = [str(k) for k in mapping.keys()]

                mapped_nodes = set(mapping.values())
                all_nodes = set(range(data.num_nodes))
                unaligned = sorted(all_nodes - mapped_nodes)
                data.unaligned_indices = torch.tensor(unaligned, dtype=torch.long)

                structures.append(data)

            self.alignment_groups.append({
                "alignment_id": alignment_id,
                "structures": structures,
            })

    def _resolve_alignment_mapping(self, alignment_id, sequence_id) -> Dict[int, int]:
        mapping: Dict[int, int] = {}
        alignment_entry = self.alignment_map.get(alignment_id, {})

        possible_keys: List[str] = []
        if sequence_id is not None:
            possible_keys.extend([
                str(sequence_id),
                f"rna_{sequence_id}",
                f"seq_{sequence_id}",
            ])

        for key in possible_keys:
            if key in alignment_entry:
                mapping = alignment_entry[key]
                break

        resolved: Dict[int, int] = {}
        for align_pos, struct_pos in mapping.items():
            try:
                align_pos_int = int(align_pos)
                struct_pos_int = int(struct_pos) - 1
            except (TypeError, ValueError):
                continue
            if struct_pos_int >= 0:
                resolved[align_pos_int] = struct_pos_int
        return resolved

    def __len__(self) -> int:
        return len(self.alignment_groups)

    def __getitem__(self, idx: int):
        return self.alignment_groups[idx]
