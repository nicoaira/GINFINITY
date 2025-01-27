import torch
import unittest
from src.model.gin_model import GINModel
from src.utils import dotbracket_to_graph, graph_to_tensor, dotbracket_to_forgi_graph, forgi_graph_to_tensor

class TestGINModel(unittest.TestCase):
    def setUp(self):
        # Set up model and dummy data for testing
        self.hidden_dim = 256
        self.output_dim = 128
        self.gin_layers = 2
        self.device = 'cpu'

        # Instantiate both standard and forgi models
        self.standard_model = GINModel(hidden_dim=self.hidden_dim, output_dim=self.output_dim, 
                                     graph_encoding="standard", gin_layers=self.gin_layers)
        self.forgi_model = GINModel(hidden_dim=self.hidden_dim, output_dim=self.output_dim, 
                                   graph_encoding="forgi", gin_layers=self.gin_layers)
        
        # Example dot-bracket structure
        self.dot_bracket_structure = "((((...))))...((((....))))"

    def test_standard_graph_conversion(self):
        # Test if the standard graph conversion works
        graph = dotbracket_to_graph(self.dot_bracket_structure)
        tensor_data = graph_to_tensor(graph)
        self.assertIsNotNone(tensor_data.x)
        self.assertIsNotNone(tensor_data.edge_index)

    def test_forgi_graph_conversion(self):
        # Test if the forgi graph conversion works
        graph = dotbracket_to_forgi_graph(self.dot_bracket_structure)
        tensor_data = forgi_graph_to_tensor(graph)
        self.assertIsNotNone(tensor_data.x)
        self.assertIsNotNone(tensor_data.edge_index)

    def test_model_forward_standard(self):
        # Test if the model generates an embedding with standard encoding
        graph = dotbracket_to_graph(self.dot_bracket_structure)
        tensor_data = graph_to_tensor(graph)
        
        with torch.no_grad():
            embedding = self.standard_model.forward_once(tensor_data)
            
        self.assertEqual(embedding.shape, torch.Size([1, self.output_dim]))

    def test_model_forward_forgi(self):
        # Test if the model generates an embedding with forgi encoding
        graph = dotbracket_to_forgi_graph(self.dot_bracket_structure)
        tensor_data = forgi_graph_to_tensor(graph)
        
        with torch.no_grad():
            embedding = self.forgi_model.forward_once(tensor_data)
            
        self.assertEqual(embedding.shape, torch.Size([1, self.output_dim]))

if __name__ == '__main__':
    unittest.main()