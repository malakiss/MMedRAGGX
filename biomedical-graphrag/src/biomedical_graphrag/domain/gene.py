from pydantic import BaseModel, Field


class GeneRecord(BaseModel):
    gene_id: str = Field(default="", description="NCBI GeneID")
    name: str = Field(default="", description="Gene symbol/name")
    description: str = Field(default="", description="Gene description or summary")
    chromosome: str = Field(default="", description="Chromosome identifier")
    map_location: str = Field(default="", description="Map location")
    organism: str = Field(default="", description="Organism scientific name")
    aliases: str = Field(default="", description="Other aliases (raw string from NCBI)")
    designations: str = Field(default="", description="Other designations (raw string from NCBI)")
    linked_pmids: list[str] = Field(default_factory=list, description="Linked PubMed PMIDs")
