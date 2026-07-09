"""One-off inspection: print retrieved chunks for specific queries."""

from __future__ import annotations

from app.opensearch.client import get_opensearch_client
from app.llm.tools import execute_search_images
from app.retrieval.service import hybrid_retrieve


def main() -> None:
    client = get_opensearch_client()

    user_id = 2
    top_k = 8

    queries = [
        "who's the chairwoman",
        "show an image of her",
    ]

    for query in queries:
        print("=" * 80)
        print("QUERY:", repr(query))
        print("-" * 80)
        response = hybrid_retrieve(client, query, user_id=user_id, top_k=top_k)
        print(f"total={response.total}")
        for index, chunk in enumerate(response.results, start=1):
            extra = chunk.extra_metadata or {}
            caption = extra.get("image_caption") or ""
            preview = chunk.content.replace("\n", " ")[:200]
            print(
                f"[{index}] score={chunk.score:.4f} type={chunk.chunk_type} "
                f"page={chunk.page_number} id={chunk.chunk_id}"
            )
            print(f"    file={chunk.filename}")
            if chunk.image_url:
                print(f"    image_url={chunk.image_url}")
            if caption:
                print(f"    caption={caption[:160]}")
            print(f"    preview={preview!r}")
        print()

    image_queries = [
        "show an image of her",
        "Meng Wanzhou portrait photo",
        "chairwoman portrait",
        "Meng Wanzhou",
    ]
    for query in image_queries:
        print("=" * 80)
        print("IMAGE-ONLY SEARCH:", repr(query))
        print("-" * 80)
        payload, images = execute_search_images(client, user_id=user_id, query=query)
        print(payload)
        for index, image in enumerate(images, start=1):
            print(
                f"  [{index}] score={image.get('score')} page={image.get('page_number')} "
                f"id={image.get('image_chunk_id')} url={image.get('image_url')} "
                f"caption={(image.get('caption') or '')[:120]}"
            )
        print()


if __name__ == "__main__":
    main()
