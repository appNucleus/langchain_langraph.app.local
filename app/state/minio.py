from __future__ import annotations

import asyncio
import io
from typing import Any, BinaryIO

from minio import Minio
from minio.error import S3Error


class MinioArtifactStore:
    """Optional object storage for large artifacts and uploaded files."""

    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        *,
        bucket: str,
        secure: bool = False,
    ) -> None:
        self.bucket = bucket
        self.client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
        )

    async def start(self) -> None:
        exists = await asyncio.to_thread(self.client.bucket_exists, self.bucket)
        if not exists:
            await asyncio.to_thread(self.client.make_bucket, self.bucket)

    async def aclose(self) -> None:
        return None

    async def health(self) -> dict[str, Any]:
        exists = await asyncio.to_thread(self.client.bucket_exists, self.bucket)
        return {
            "status": "available" if exists else "unavailable",
            "backend": "minio",
            "bucket": self.bucket,
        }

    async def put_bytes(
        self,
        object_name: str,
        data: bytes,
        *,
        content_type: str = "application/octet-stream",
    ) -> str:
        stream: BinaryIO = io.BytesIO(data)
        await asyncio.to_thread(
            self.client.put_object,
            self.bucket,
            object_name,
            stream,
            len(data),
            content_type=content_type,
        )
        return object_name

    async def get_bytes(self, object_name: str) -> bytes:
        response = None
        try:
            response = await asyncio.to_thread(
                self.client.get_object, self.bucket, object_name
            )
            return await asyncio.to_thread(response.read)
        except S3Error:
            raise
        finally:
            if response is not None:
                response.close()
                response.release_conn()
