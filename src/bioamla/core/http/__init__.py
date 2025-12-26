"""
HTTP Client Utilities
=====================

Provides reusable HTTP client functionality for interfacing with external APIs.

Features:
- Thread-safe rate limiting with configurable requests per second
- Automatic retry with exponential backoff
- Unified HTTP client with timeout and error handling
- Disk-based caching with TTL for API responses
- @config_aware decorator for configuration-driven API methods
"""

from .client import (
    APICache,
    APIClient,
    ConfigAwareMixin,
    RateLimiter,
    clear_cache,
    config_aware,
    config_aware_class,
    get_cache_dir,
    rate_limited,
)

__all__ = [
    "APICache",
    "APIClient",
    "ConfigAwareMixin",
    "RateLimiter",
    "clear_cache",
    "config_aware",
    "config_aware_class",
    "get_cache_dir",
    "rate_limited",
]
