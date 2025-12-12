from pydantic import BaseModel, Field

from biomedical_graphrag.domain.citation import CitationNetwork
from biomedical_graphrag.domain.gene import GeneRecord
from biomedical_graphrag.domain.paper import Paper


class PaperMetadata(BaseModel):
    collection_date: str = Field(default="", description="Date of data collection")
    query: str = Field(default="", description="Query used for data collection")
    total_papers: int = Field(default=0, description="Total number of papers")
    papers_with_citations: int = Field(default=0, description="Number of papers with citations")
    total_authors: int = Field(default=0, description="Total number of authors")
    total_mesh_terms: int = Field(default=0, description="Total number of MeSH terms")


class PaperDataset(BaseModel):
    metadata: PaperMetadata = Field(default_factory=PaperMetadata, description="Dataset metadata")
    papers: list[Paper] = Field(default_factory=list, description="List of papers in the dataset")
    citation_network: dict[str, CitationNetwork] = Field(
        default_factory=dict,
        description="Citation network mapping paper IDs to their citation details",
    )


# Gene dataset models
class GeneMetadata(BaseModel):
    collection_date: str = Field(default="", description="Date of data collection")
    total_genes: int = Field(default=0, description="Total number of genes")
    genes_with_pubmed_links: int = Field(default=0, description="Genes with at least one linked PMID")
    total_linked_pmids: int = Field(
        default=0, description="Total count of linked PMIDs across all genes"
    )


class GeneDataset(BaseModel):
    metadata: GeneMetadata = Field(default_factory=GeneMetadata, description="Gene dataset metadata")
    genes: list[GeneRecord] = Field(default_factory=list, description="List of genes with PubMed links")
