from biomedical_graphrag.infrastructure.neo4j_db.neo4j_client import AsyncNeo4jClient


async def delete_graph() -> None:
    """
    Delete all nodes and relationships in the Neo4j graph database.

    Args:
        None
    Returns:
        None
    """

    client = await AsyncNeo4jClient.create()
    await client.delete_graph()
    await client.close()


if __name__ == "__main__":
    import asyncio

    asyncio.run(delete_graph())
