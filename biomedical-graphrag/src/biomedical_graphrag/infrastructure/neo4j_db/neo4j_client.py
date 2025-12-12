from typing import Any

from neo4j import AsyncGraphDatabase

from biomedical_graphrag.config import settings


class AsyncNeo4jClient:
    """
    Async Neo4j client for creating and deleting graphs.
    """

    def __init__(self, driver: Any) -> None:
        self.driver = driver
        self.database = settings.neo4j.database

    @classmethod
    async def create(cls) -> "AsyncNeo4jClient":
        """
        Construct an async driver using AsyncGraphDatabase.
        """
        uri = settings.neo4j.uri
        user = settings.neo4j.username
        password = settings.neo4j.password.get_secret_value()

        driver = AsyncGraphDatabase.driver(
            uri,
            auth=(user, password),
            max_connection_lifetime=300,  # seconds
            connection_acquisition_timeout=60,
            connection_timeout=30,
            keep_alive=True,
        )
        return cls(driver)

    async def close(self) -> None:
        """
        Close the async Neo4j driver.
        """
        await self.driver.close()

    async def execute(self, cypher_query: str, parameters: dict[str, Any] | None = None) -> None:
        """
        Execute a Cypher query without returning records.
        """
        async with self.driver.session(database=self.database) as session:
            await session.run(cypher_query, parameters or {})

    async def create_graph(self, cypher_query: str, parameters: dict[str, Any] | None = None) -> None:
        """
        Execute a Cypher query to create nodes and relationships in the graph.
        """
        await self.execute(cypher_query, parameters)

    async def delete_graph(self) -> None:
        """
        Delete all nodes and relationships in the graph.
        """
        await self.execute("MATCH (n) DETACH DELETE n")
