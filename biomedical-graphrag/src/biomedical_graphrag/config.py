import os
from typing import ClassVar

from pydantic import BaseModel, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from biomedical_graphrag.utils.logger_util import setup_logging

logger = setup_logging()


class GeminiSettings(BaseModel):
    api_key: SecretStr = Field(default=SecretStr(""), description="API key for Google Gemini")
    model: str = Field(default="gemini-2.5-flash", description="Gemini model to use for queries")
    temperature: float = Field(
        default=0.0, description="LLM temperature for Gemini queries (0 for consistency)"
    )
    embedding_model: str = Field(default="text-embedding-004", description="Gemini embedding model for vectors")
    embedding_dimension: int = Field(default=768, description="Dimension of the Gemini embedding model (768 for text-embedding-004)")
    max_tokens: int = Field(default=1500, description="Maximum number of tokens for Gemini queries")


class Neo4jSettings(BaseModel):
    uri: str = Field(default="bolt://localhost:7687", description="URI for Neo4j database")
    username: str = Field(default="neo4j", description="Username for Neo4j database")
    password: SecretStr = Field(default=SecretStr(""), description="Password for Neo4j database")
    database: str = Field(default="neo4j", description="Database name for Neo4j database")


# class QdrantSettings(BaseModel):
#     url: str = Field(default="http://localhost:6333", description="URL for Qdrant instance")
#     api_key: SecretStr = Field(default=SecretStr(""), description="API key for Qdrant instance")
#     collection_name: str = Field(
#         default="biomedical_papers", description="Collection name for Qdrant instance"
#     )
#     embedding_model: str = Field(
#         default="text-embedding-3-small", description="OpenAI embedding model to use"
#     )
#     embedding_dimension: int = Field(default=1536, description="Dimension of the OpenAI embedding model")

class QdrantSettings(BaseModel):
    url: str = Field(default="http://localhost:6333", description="URL for Qdrant instance")
    api_key: SecretStr = Field(default=SecretStr(""), description="API key for Qdrant instance")
    collection_name: str = Field(
        default="biomedical_radiology", description="Collection name for biomedical papers and radiology"
    )
    embedding_model: str = Field(
        default="text-embedding-004", description="Gemini embedding model to use"
    )
    embedding_dimension: int = Field(default=768, description="Dimension of the Gemini embedding model (768 for text-embedding-004)")

class PubMedSettings(BaseModel):
    email: SecretStr = Field(default=SecretStr(""), description="Email for PubMed API")
    api_key: SecretStr = Field(default=SecretStr(""), description="API key for PubMed API")


class JsonDataSettings(BaseModel):
    pubmed_json_path: str = Field(
        default="data/pubmed_dataset.json", description="Path to the PubMed JSON dataset"
    )
    radiology_json_path: str = Field(
        default="/home/m.ismail/MMed-RAG/data/training/retriever/radiology/rad_iu.json",
        description="Path to the IU X-Ray radiology dataset JSON"
    )


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    Use .env file or environment variables to configure.
    """

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_file=[".env"],
        env_file_encoding="utf-8",
        extra="ignore",
        env_nested_delimiter="__",
        case_sensitive=False,
        frozen=True,
    )

    gemini: GeminiSettings = GeminiSettings()
    neo4j: Neo4jSettings = Neo4jSettings()
    qdrant: QdrantSettings = QdrantSettings()
    pubmed: PubMedSettings = PubMedSettings()
    json_data: JsonDataSettings = JsonDataSettings()

    @model_validator(mode="after")
    def validate_json_path(self) -> "Settings":
        """Validate that the JSON data path exists."""
        if not os.path.isfile(self.json_data.pubmed_json_path):
            logger.warning(
                f"PubMed JSON file not found at {self.json_data.pubmed_json_path}. "
                "Proceeding without it."
            )
        return self

    @model_validator(mode="after")
    def validate_radiology_json_path(self) -> "Settings":
        """Validate that the Radiology JSON data path exists."""
        if not os.path.isfile(self.json_data.radiology_json_path):
            logger.warning(
                f"Radiology JSON file not found at {self.json_data.radiology_json_path}. Proceeding without it."
            )
        return self


settings = Settings()
