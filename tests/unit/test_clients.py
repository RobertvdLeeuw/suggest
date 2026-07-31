# We never exceeds the spotify rate limit.

# API Tokens are always refreshed before expiration.

# API calls that failes due to networking (not 404 or smth) are handled with exponential backoff.

# API calls that return paginated data are always collected in their entirety.
