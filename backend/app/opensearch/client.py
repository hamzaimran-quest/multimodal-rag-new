"""OpenSearch client factory."""

from opensearchpy import OpenSearch

from app.config import settings


def get_opensearch_client() -> OpenSearch:
    return OpenSearch(
        hosts=[{"host": settings.opensearch_host, "port": settings.opensearch_port}],
        use_ssl=False,
        verify_certs=False,
        ssl_show_warn=False,
    )
