import asyncio

from Bio import Entrez

from biomedical_graphrag.utils.logger_util import setup_logging

logger = setup_logging()


class PubMedAPIClient:
    """
    PubMed E-utilities API client for fetching paper IDs, details, and citations (async).
    """

    def __init__(self) -> None:
        # Entrez globals configured by collectors via BaseDataSource
        ...

    async def search(self, query: str, max_results: int = 100, sort: str = "relevance") -> list[str]:
        """
        Search for papers in the PubMed database (async).
        Args:
            query (str): The search query.
            max_results (int): The maximum number of results to return.
            sort (str): The sort order of the results.
        Returns:
            list[str]: A list of PubMed IDs (PMIDs) matching the search query.
        """
        # rate limiting handled by collector
        # Debug log types of all arguments
        logger.info(
            f"search() argument types: query={type(query)}, \
            max_results={type(max_results)}, sort={type(sort)}"
        )
        # Enforce correct types
        query = str(query)
        max_results = int(max_results)
        sort = str(sort)
        # Check and log Entrez.api_key and Entrez.email types (for debugging)
        logger.info(f"Entrez.api_key type before esearch: {type(getattr(Entrez, 'api_key', None))}")
        logger.info(f"Entrez.email type before esearch: {type(getattr(Entrez, 'email', None))}")

        def _search() -> list[str]:
            handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort=sort)
            record = Entrez.read(handle)
            pmids = record.get("IdList", [])
            handle.close()
            return pmids

        return await asyncio.to_thread(_search)

    async def fetch_papers(self, pmid_list: list[str]) -> list[dict]:
        """
        Fetch paper details from PubMed using a list of PMIDs (async).
        Args:
            pmid_list (list[str]): A list of PubMed IDs (PMIDs) to fetch.
        Returns:
            list[dict]: A list of dictionaries containing paper details.
        """
        if not pmid_list:
            return []

        def _fetch() -> list[dict]:
            ids = ",".join(pmid_list)
            handle = Entrez.efetch(db="pubmed", id=ids, rettype="medline", retmode="xml")
            records = Entrez.read(handle)
            handle.close()
            return records.get("PubmedArticle", [])

        return await asyncio.to_thread(_fetch)

    async def fetch_citations(self, pmid: str) -> dict[str, list[str]]:
        """
        Fetch citations for a given PubMed ID (PMID) (async).
        Args:
            pmid (str): The PubMed ID to fetch citations for.
        Returns:
            dict[str, list[str]]: A dictionary containing lists of cited by and references PMIDs.
        """

        def _fetch_citations() -> dict[str, list[str]]:
            result: dict[str, list[str]] = {"cited_by": [], "references": []}
            handle = Entrez.elink(
                dbfrom="pubmed", db="pubmed", id=pmid, linkname="pubmed_pubmed_citedin"
            )
            citing_record = Entrez.read(handle)
            handle.close()
            if citing_record[0].get("LinkSetDb"):
                result["cited_by"] = [link["Id"] for link in citing_record[0]["LinkSetDb"][0]["Link"]]
            handle = Entrez.elink(dbfrom="pubmed", db="pubmed", id=pmid, linkname="pubmed_pubmed_refs")
            refs_record = Entrez.read(handle)
            handle.close()
            if refs_record[0].get("LinkSetDb"):
                result["references"] = [link["Id"] for link in refs_record[0]["LinkSetDb"][0]["Link"]]
            return result

        return await asyncio.to_thread(_fetch_citations)
