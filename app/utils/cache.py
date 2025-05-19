"""
Simple in-memory cache implementation.
"""
from typing import Any, Dict, Optional
import time

class Cache:
    def __init__(self, ttl: int = 3600):
        """
        Initialize cache with TTL in seconds.
        
        Args:
            ttl: Time to live for cache entries in seconds (default: 1 hour)
        """
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._ttl = ttl
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get a value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        if key not in self._cache:
            return None
            
        entry = self._cache[key]
        if time.time() > entry["expires_at"]:
            del self._cache[key]
            return None
            
        return entry["value"]
    
    def set(self, key: str, value: Dict[str, Any], ttl: Optional[int] = None) -> None:
        """
        Set a value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional TTL override in seconds
        """
        self._cache[key] = {
            "value": value,
            "expires_at": time.time() + (ttl or self._ttl)
        }
    
    def delete(self, key: str) -> None:
        """
        Delete a value from cache.
        
        Args:
            key: Cache key
        """
        if key in self._cache:
            del self._cache[key]
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear() 