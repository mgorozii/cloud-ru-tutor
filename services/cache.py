import hashlib
import pickle
from datetime import datetime, timedelta

class ResponseCache:
    def __init__(self, ttl_hours: int = 24):
        self.cache = {}
        self.ttl = timedelta(hours=ttl_hours)
    
    def get(self, query: str, subject: str = None):
        key = self._generate_key(query, subject)
        if key in self.cache:
            entry = self.cache[key]
            if datetime.now() - entry["timestamp"] < self.ttl:
                return entry["response"]
        return None
    
    def set(self, query: str, response: str, subject: str = None):
        key = self._generate_key(query, subject)
        self.cache[key] = {
            "response": response,
            "timestamp": datetime.now()
        }