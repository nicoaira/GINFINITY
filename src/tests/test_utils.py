import pytest
from ..utils import is_valid_dot_bracket, center_pad_matrix, dotbracket_to_graph




def test_is_valid_dot_bracket():
    # Valid structures
    
    # URS00000478B7
    assert is_valid_dot_bracket("(((((((((((....))))))))....(((..........)))))).((((((((((.((((((((((((((((....(((((((((((((.(((((((...((((((((((.(((((((((((((((((((((((((((((((((....))))))))))))))))))((...))(((((((...(((((....(((....)))....))))).))))))))))))))))).)))))))...)))))))))))))))))))))))))))))))))))))..))))))))))))))))).") == True

    # URS000075D68F
    assert is_valid_dot_bracket(".((((................................((((((...((((((((...........(((((((((((.............(((((.((((.........)))))))))........))))))).......(.((((((.......(((((((((.......)..)))))))).......)))))))..................)))).......))))))))....................))))))...............)))).") == True

    # URS000075B5A7
    assert is_valid_dot_bracket(".(((((((..(((((...((((((((((((((..((((..((((.(..((((..(.((((.....)))))..))))..))))).....))))..))))))((((((.....)))))).....)))))))).))))).....))))))).............(((((..(.((((......((((((((...(((.(.(((.......)))....).)))))))))))....)))).)..)))))......") == True

    # URS00006B41C9
    assert is_valid_dot_bracket("(((((((((((..(((((.....))))).)))))).))))).......(((.((......))))).........(((((..............((((((((((((..(((.......((((...(((((((((......))))).))))..)))).........))).(((((((((....))))))).))..)))))))))))).......)))))") == True

    # Invalid structures
    assert is_valid_dot_bracket("(") == False
    assert is_valid_dot_bracket(")") == False
    assert is_valid_dot_bracket("((.)") == False
    assert is_valid_dot_bracket("(.))") == False
    assert is_valid_dot_bracket(")(") == False

    # Modified real structures with errors
    # URS00000478B7 - unmatched parenthesis
    assert is_valid_dot_bracket("(((((((((((....))))))))....(((..........)))))).((((((((((.((((((((((((((((....(((((((((((((.(((((((...((((((((((.(((((((((((((((((((((((((((((((((....))))))))))))))))))((...))(((((((...(((((....(((....)))....))))).))))))))))))))))).)))))))...)))))))))))))))))))))))))))))))))))))..))))))))))))))))))") == False

    # URS000075D68F - crossed brackets
    assert is_valid_dot_bracket(".((((................................((((((...((((((((...........(((((((((((.............(((((.((((.........)))))))))........))))))).......(.((((((.......(((((((((.......)..)))))))).......)))))))(..................)))).......))))))))....................))))))...............))))(") == False

    # URS000075B5A7 - missing closing bracket
    assert is_valid_dot_bracket(".(((((((..(((((...((((((((((((((..((((..((((.(..((((..(.((((.....)))))..))))..))))).....))))..))))))((((((.....)))))).....)))))))).))))).....)))))).............(((((..(.((((......((((((((...(((.(.(((.......)))....).)))))))))))....)))).)..)))))....(") == False

    # Empty structure
    assert is_valid_dot_bracket("") == True

def test_center_pad_matrix():
    # Test even padding
    assert center_pad_matrix("((()))", 8) == ".((()))."
    
    # Test odd padding
    assert center_pad_matrix("(())", 7) == ".(()).."
    
    # Test no padding needed
    assert center_pad_matrix("(())", 4) == "(())"
    
    # Test empty structure
    assert center_pad_matrix("", 4) == "...."
    
    # Test custom padding value
    assert center_pad_matrix("(())", 8, padding_value='*') == "**(())**"
    
    # Test single character
    assert center_pad_matrix(".", 5) == "....."
    
    # Test longer dot-bracket
    assert center_pad_matrix("(((...)))", 15) == "...(((...)))..."

def test_dotbracket_to_graph():
    # Test empty structure
    assert dotbracket_to_graph("") is not None

    # Test single unpaired base
    g = dotbracket_to_graph(".")
    assert len(g.nodes) == 1
    assert g.nodes[0]["label"] == "unpaired"
    assert len(g.edges) == 0

    # Test simple hairpin
    g = dotbracket_to_graph("((...))")
    assert len(g.nodes) == 7
    assert g.nodes[0]["label"] == "paired"
    assert g.nodes[1]["label"] == "paired" 
    assert g.nodes[2]["label"] == "unpaired"
    assert g.nodes[3]["label"] == "unpaired"
    assert g.nodes[4]["label"] == "unpaired"
    assert g.nodes[5]["label"] == "paired"
    assert g.nodes[6]["label"] == "paired"
    
    # Check base pair edges
    assert (0,6) in g.edges
    assert (1,5) in g.edges
        
    # Check adjacent edges
    assert (0,1) in g.edges
    assert (1,2) in g.edges
    assert (2,3) in g.edges
    assert (3,4) in g.edges
    assert (4,5) in g.edges
    assert (5,6) in g.edges
