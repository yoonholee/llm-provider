"""Rate-limit header parsing for adaptive concurrency control."""


def parse_openai_headers(headers) -> dict | None:
    """Extract rate-limit info from OpenAI response headers.

    Returns {"remaining": int, "limit": int} or None if headers not present.

    OpenAI headers:
        x-ratelimit-remaining-requests
        x-ratelimit-limit-requests
    """
    remaining = headers.get("x-ratelimit-remaining-requests")
    limit = headers.get("x-ratelimit-limit-requests")
    if remaining is not None and limit is not None:
        try:
            r, l = int(remaining), int(limit)
            if l > 0:
                return {"remaining": r, "limit": l}
        except (ValueError, TypeError):
            pass
    return None
