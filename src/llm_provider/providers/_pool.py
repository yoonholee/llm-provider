"""Generic round-robin client pool for API key rotation.

Usage:
    GEMINI_API_KEY=key1,key2,key3
    OPENAI_API_KEY=key1,key2
    TOGETHER_API_KEY=key1,key2

On 429, the provider calls pool.rotate() to advance to the next key.
"""

import threading


class ClientPool:
    """Round-robin pool of API clients.

    Wraps a list of clients created from comma-separated API keys.
    Thread-safe rotation on 429 errors.
    """

    def __init__(self, clients: list):
        if not clients:
            raise ValueError("At least one client required")
        self._clients = clients
        self._idx = 0
        self._lock = threading.Lock()

    @property
    def current(self):
        return self._clients[self._idx % len(self._clients)]

    def rotate(self) -> bool:
        """Advance to the next client. Returns True if there are multiple keys."""
        if len(self._clients) <= 1:
            return False
        with self._lock:
            self._idx = (self._idx + 1) % len(self._clients)
        return True

    def __len__(self):
        return len(self._clients)

    def close_all(self):
        """Close all clients in the pool."""
        for client in self._clients:
            if hasattr(client, "close"):
                try:
                    client.close()
                except TypeError:
                    pass  # async close — caller handles via aclose_all()
                except Exception:
                    pass

    async def aclose_all(self):
        """Close all async clients in the pool."""
        for client in self._clients:
            if hasattr(client, "close"):
                try:
                    result = client.close()
                    if hasattr(result, "__await__"):
                        await result
                except Exception:
                    pass

    # Allow pool to be used directly as the client (single-key case)
    def __getattr__(self, name):
        return getattr(self.current, name)
