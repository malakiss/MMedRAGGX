"""Unit tests for domain models."""

from biomedical_graphrag.domain.author import Author
from biomedical_graphrag.domain.meshterm import MeSHTerm
from biomedical_graphrag.domain.paper import Paper


def test_paper_creation(sample_paper: Paper) -> None:
    """Test that a Paper can be created with valid data."""
    assert sample_paper.pmid == "12345678"
    assert sample_paper.title == "Test Paper Title"
    assert sample_paper.abstract == "This is a test abstract for the paper."
    assert len(sample_paper.authors) == 1
    assert len(sample_paper.mesh_terms) == 1
    assert sample_paper.publication_date == "2024-01-01"
    assert sample_paper.journal == "Test Journal"
    assert sample_paper.doi == "10.1000/test.doi"


def test_paper_with_empty_data() -> None:
    """Test that a Paper can be created with empty/default data."""
    paper = Paper()
    assert paper.pmid == ""
    assert paper.title == ""
    assert paper.abstract == ""
    assert paper.authors == []
    assert paper.mesh_terms == []
    assert paper.publication_date == ""
    assert paper.journal == ""
    assert paper.doi == ""


def test_author_creation(sample_author: Author) -> None:
    """Test that an Author can be created with valid data."""
    assert sample_author.name == "John Doe"
    assert sample_author.first_name == "John"
    assert sample_author.last_name == "Doe"
    assert len(sample_author.affiliations) == 2
    assert "Test University" in sample_author.affiliations
    assert "Research Institute" in sample_author.affiliations


def test_mesh_term_creation(sample_mesh_term: MeSHTerm) -> None:
    """Test that a MeSHTerm can be created with valid data."""
    assert sample_mesh_term.term == "Test Term"
    assert sample_mesh_term.ui == "D123456"
    assert sample_mesh_term.major_topic is True
    assert len(sample_mesh_term.qualifiers) == 2
    assert "therapy" in sample_mesh_term.qualifiers
    assert "diagnosis" in sample_mesh_term.qualifiers
