from pydantic import BaseModel, Field


class CitationNetwork(BaseModel):
    pmid: str = Field(default="", description="PubMed ID of the paper")
    cited_by: list[str] = Field(
        default_factory=list, description="List of PubMed IDs that cite this paper"
    )
    references: list[str] = Field(
        default_factory=list, description="List of PubMed IDs referenced by this paper"
    )
