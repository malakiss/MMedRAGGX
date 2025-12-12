from pydantic import BaseModel, Field

from biomedical_graphrag.domain.author import Author
from biomedical_graphrag.domain.meshterm import MeSHTerm


class Paper(BaseModel):
    pmid: str = Field(default="", description="PubMed ID of the paper")
    title: str = Field(default="", description="Title of the paper")
    abstract: str = Field(default="", description="Abstract of the paper")
    authors: list[Author] = Field(
        default_factory=list[Author], description="List of authors with their details"
    )
    mesh_terms: list[MeSHTerm] = Field(
        default_factory=list[MeSHTerm], description="MeSH terms associated with the paper"
    )
    publication_date: str = Field(default="", description="Publication date of the paper")
    journal: str = Field(default="", description="Journal where the paper was published")
    doi: str = Field(default="", description="Digital Object Identifier of the paper")
