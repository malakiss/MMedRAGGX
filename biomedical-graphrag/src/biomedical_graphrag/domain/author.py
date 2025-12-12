from pydantic import BaseModel, Field


class Author(BaseModel):
    """Author information"""

    name: str = Field(default="", description="Full name of the author")
    first_name: str = Field(default="", description="First name")
    last_name: str = Field(default="", description="Last name")
    affiliations: list[str] = Field(
        default_factory=list[str], description="List of institutional affiliations"
    )
