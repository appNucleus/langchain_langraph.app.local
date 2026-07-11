from __future__ import annotations
import asyncio
from collections import OrderedDict
from time import monotonic
from typing import Any

class BoundedInMemoryStore:
    def __init__(self, *, ttl_seconds: int, max_sessions: int, max_messages: int) -> None:
        self.ttl=ttl_seconds; self.max_sessions=max_sessions; self.max_messages=max_messages
        self._data: OrderedDict[str, tuple[float,list[dict[str,Any]]]]=OrderedDict(); self._lock=asyncio.Lock()
    async def get(self, key: str) -> list[dict[str,Any]]:
        async with self._lock:
            self._purge(); item=self._data.get(key)
            if not item: return []
            _, messages=item; self._data.move_to_end(key); return [dict(m) for m in messages]
    async def append(self, key: str, *messages: dict[str,Any]) -> None:
        async with self._lock:
            self._purge(); current=self._data.get(key,(0,[]))[1]
            current=(current+[dict(m) for m in messages])[-self.max_messages:]
            self._data[key]=(monotonic(),current); self._data.move_to_end(key)
            while len(self._data)>self.max_sessions: self._data.popitem(last=False)
    def _purge(self) -> None:
        cutoff=monotonic()-self.ttl
        for key,(ts,_) in list(self._data.items()):
            if ts<cutoff: self._data.pop(key,None)
