import argparse
import asyncio

from biomedical_graphrag.application.services.query_vectorstore_service.qdrant_query import (
    AsyncQdrantQuery,
)
from biomedical_graphrag.utils.logger_util import setup_logging

logger = setup_logging()


async def main() -> None:
    """
    Command-line interface to query the biomedical vector store (async).
    Args:
        --ask: The question to ask (default: example question)

    Returns:
        Prints the answer to the console.

    """
    parser = argparse.ArgumentParser(description="Query the biomedical vector store.")

    parser.add_argument(
        "--ask",
        type=str,
        default=(
            "Which institutions have collaborated most"
            "frequently on papers about 'Gene Editing' and 'Immunotherapy'?"
        ),
        help="The question to ask.",
    )

    args = parser.parse_args()

    # Initialize AsyncQdrantQuery
    qdrant_query = AsyncQdrantQuery()
    try:
        # Perform the query
        answer = await qdrant_query.get_answer(args.ask)
        logger.info(f"Answer: {answer}")
    finally:
        await qdrant_query.close()


if __name__ == "__main__":
    asyncio.run(main())
