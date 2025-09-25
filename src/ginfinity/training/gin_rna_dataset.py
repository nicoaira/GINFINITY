from torch.utils.data import Dataset
from ginfinity.utils import dotbracket_to_graph, graph_to_tensor
import pandas as pd
import torch
from tqdm import tqdm
from typing import Dict, List, Any, Tuple, Optional


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

        g_anchor = dotbracket_to_graph(anchor_structure, anchor_seq, graph_encoding=self.graph_encoding)
        g_positive = dotbracket_to_graph(positive_structure, positive_seq, graph_encoding=self.graph_encoding)
        g_negative = dotbracket_to_graph(negative_structure, negative_seq, graph_encoding=self.graph_encoding)

        data_anchor = graph_to_tensor(g_anchor, self.seq_weight, graph_encoding=self.graph_encoding)
        data_positive = graph_to_tensor(g_positive, self.seq_weight, graph_encoding=self.graph_encoding)
        data_negative = graph_to_tensor(g_negative, self.seq_weight, graph_encoding=self.graph_encoding)

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

        g_anchor = dotbracket_to_graph(anchor_structure, anchor_seq, graph_encoding=self.graph_encoding)
        g_positive = dotbracket_to_graph(positive_structure, positive_seq, graph_encoding=self.graph_encoding)

        data_anchor = graph_to_tensor(g_anchor, self.seq_weight, graph_encoding=self.graph_encoding)
        data_positive = graph_to_tensor(g_positive, self.seq_weight, graph_encoding=self.graph_encoding)

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
        show_progress: bool = False,
        progress_desc: str = "Preparing alignment dataset",
        cache_preprocessed: bool = False,
    ):
        if isinstance(dataframe, str):
            df = pd.read_csv(dataframe, comment="#")
        else:
            df = dataframe.copy()

        self.graph_encoding = graph_encoding
        self.seq_weight = seq_weight
        self.alignment_groups: List[Dict[str, Any]] = []
        self.alignment_map = alignment_map
        self._progress_enabled = show_progress
        self._progress_desc = progress_desc
        self.structure_column = structure_column
        self.cache_preprocessed = cache_preprocessed
        self._cache: Optional[Dict[int, Tuple[str, List[Any]]]] = {} if cache_preprocessed else None

        # Category mapping
        self.category_to_id = {
            "5-paired": 0,
            "3-paired": 1, 
            "unpaired": 2,
            "unaligned-5-paired": 3,
            "unaligned-3-paired": 4,
            "unaligned-unpaired": 5
        }

        total_structures = len(df)
        progress = None
        if self._progress_enabled:
            progress = tqdm(
                total=total_structures,
                desc=self._progress_desc,
                unit="structure",
            )

        for alignment_id, group_df in df.groupby("alignment_id", sort=False):
            rows: List[Dict[str, Any]] = []
            for row in group_df.itertuples(index=False):
                row_data = row._asdict()
                rows.append(row_data)
                if progress is not None:
                    progress.update(1)

            self.alignment_groups.append({
                "alignment_id": alignment_id,
                "rows": rows,
            })

        if progress is not None:
            progress.close()


    def _resolve_structured_alignment_mapping(self, alignment_id, sequence_id):
        """Parse both old and new JSON formats with categories."""
        mapping: Dict[int, int] = {}
        categories: Dict[int, int] = {}  # node_idx -> category_id
        unaligned_nodes = []
        
        alignment_entry = self.alignment_map.get(alignment_id, {})

        possible_keys: List[str] = []
        if sequence_id is not None:
            possible_keys.extend([
                str(sequence_id),
                f"rna_{sequence_id}",
                f"seq_{sequence_id}",
            ])

        rna_data = None
        for key in possible_keys:
            if key in alignment_entry:
                rna_data = alignment_entry[key]
                break
        
        if rna_data is None:
            return mapping, categories, unaligned_nodes

        # Check if this is the old format (direct position -> alignment mapping)
        # or new format (categories with position mappings)
        is_old_format = self._is_old_format(rna_data)
        
        if is_old_format:
            # Handle old format: {align_pos: struct_pos, ...}
            for align_pos_str, struct_pos in rna_data.items():
                try:
                    align_pos_int = int(align_pos_str)
                    struct_pos_int = int(struct_pos) - 1  # Convert to 0-based indexing
                except (TypeError, ValueError):
                    continue
                    
                if struct_pos_int >= 0:
                    mapping[align_pos_int] = struct_pos_int
                    # Default to unpaired category for old format
                    categories[struct_pos_int] = 2  # unpaired
        else:
            # Handle new format: {category: {struct_pos: align_pos, ...}, ...}
            for category_name, category_positions in rna_data.items():
                if category_name not in self.category_to_id:
                    continue
                    
                category_id = self.category_to_id[category_name]
                is_conserved = category_id < 3  # 0, 1, 2 are conserved categories
                
                for struct_pos_str, align_pos in category_positions.items():
                    try:
                        struct_pos = int(struct_pos_str) - 1  # Convert to 0-based indexing
                        align_pos_int = int(align_pos)
                    except (TypeError, ValueError):
                        continue
                        
                    if struct_pos >= 0:
                        categories[struct_pos] = category_id
                        
                        if is_conserved:
                            # Only conserved positions go into alignment mapping
                            mapping[align_pos_int] = struct_pos
                        else:
                            # Unaligned positions are tracked separately
                            unaligned_nodes.append(struct_pos)

        return mapping, categories, sorted(unaligned_nodes)
    
    def _is_old_format(self, rna_data):
        """Check if the RNA data is in old format (direct mappings) or new format (categorized)."""
        if not isinstance(rna_data, dict):
            return False
            
        # Check if all keys are numeric strings (old format)
        # or if any key matches our category names (new format)
        for key in rna_data.keys():
            if key in self.category_to_id:
                return False  # Found a category name, this is new format
        
        # If no category names found, assume old format
        return True

    def __len__(self) -> int:
        return len(self.alignment_groups)

    def __getitem__(self, idx: int):
        if self.cache_preprocessed and self._cache is not None and idx in self._cache:
            alignment_id, cached_structures = self._cache[idx]
            return {
                "alignment_id": alignment_id,
                "structures": [data.clone() for data in cached_structures],
            }

        alignment_entry = self.alignment_groups[idx]
        alignment_id = alignment_entry["alignment_id"]
        rows = alignment_entry["rows"]
        structures: List[Any] = []

        for row_data in rows:
            structure = row_data[self.structure_column]
            sequence = row_data.get("sequence")
            g = dotbracket_to_graph(structure, sequence, graph_encoding=self.graph_encoding)
            data = graph_to_tensor(g, self.seq_weight, graph_encoding=self.graph_encoding)

            sequence_id = row_data.get("sequence_id")
            if sequence_id is not None and pd.notna(sequence_id):
                try:
                    sequence_id_int = int(sequence_id)
                except (TypeError, ValueError):
                    sequence_id_int = sequence_id
            else:
                sequence_id_int = None

            data.sequence_id = sequence_id_int
            data.alignment_id = alignment_id
            data.binary_code = row_data.get("binary_code")

            mapping, categories, unaligned = self._resolve_structured_alignment_mapping(
                alignment_id,
                sequence_id_int,
            )

            mapping, categories, unaligned = self._filter_alignment_annotations(
                mapping,
                categories,
                unaligned,
                data.num_nodes,
            )

            data._alignment_mapping = mapping

            num_nodes = data.num_nodes
            node_categories_tensor = torch.full((num_nodes,), 5, dtype=torch.long)
            for node_idx, category_id in categories.items():
                if 0 <= node_idx < num_nodes:
                    node_categories_tensor[node_idx] = category_id

            data.node_categories = node_categories_tensor
            data.unaligned_indices = torch.tensor(unaligned, dtype=torch.long)

            structures.append(data)

        result = {
            "alignment_id": alignment_id,
            "structures": structures,
        }

        if self.cache_preprocessed and self._cache is not None:
            self._cache[idx] = (
                alignment_id,
                [data.clone() for data in structures],
            )

        return result

    def _filter_alignment_annotations(
        self,
        mapping,
        categories,
        unaligned_nodes,
        graph_size: int,
    ):
        """Drop alignment metadata that points outside the current graph."""

        if graph_size <= 0:
            return {}, {}, []

        if mapping:
            mapping = {
                align_pos: struct_pos
                for align_pos, struct_pos in mapping.items()
                if 0 <= struct_pos < graph_size
            }
        else:
            mapping = {}

        if categories:
            categories = {
                node_idx: category_id
                for node_idx, category_id in categories.items()
                if 0 <= node_idx < graph_size
            }
        else:
            categories = {}

        if unaligned_nodes:
            unaligned_nodes = [
                node_idx for node_idx in unaligned_nodes if 0 <= node_idx < graph_size
            ]
        else:
            unaligned_nodes = []

        return mapping, categories, unaligned_nodes
