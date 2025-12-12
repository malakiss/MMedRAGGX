from pydantic import BaseModel, Field


class MeSHTerm(BaseModel):
    """Medical Subject Heading term"""

    term: str = Field(default="", description="MeSH term text")
    ui: str = Field(default="", description="Unique identifier for the MeSH term")
    major_topic: bool = Field(default=False, description="Whether this is a major topic for the paper")
    qualifiers: list[str] = Field(default_factory=list, description="MeSH qualifiers/sub-topics")
