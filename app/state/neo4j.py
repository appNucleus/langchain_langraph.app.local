from __future__ import annotations

from typing import Any

from neo4j import AsyncDriver, AsyncGraphDatabase


class Neo4jConnectionManager:
    """Own one application-scoped async Neo4j driver and its bounded pool."""

    def __init__(
        self,
        uri: str,
        username: str,
        password: str,
        *,
        database: str = "neo4j",
        max_connection_pool_size: int = 20,
        connection_acquisition_timeout: float = 30.0,
        connection_timeout: float = 10.0,
        max_connection_lifetime: float = 3600.0,
        keep_alive: bool = True,
    ) -> None:
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self.max_connection_pool_size = max_connection_pool_size
        self.connection_acquisition_timeout = connection_acquisition_timeout
        self.connection_timeout = connection_timeout
        self.max_connection_lifetime = max_connection_lifetime
        self.keep_alive = keep_alive
        self._driver: AsyncDriver | None = None

    @property
    def driver(self) -> AsyncDriver:
        """Return the shared driver after startup."""

        if self._driver is None:
            raise RuntimeError("Neo4j connection manager is not started")
        return self._driver

    async def start(self) -> None:
        """Create and verify the driver exactly once."""

        if self._driver is not None:
            return
        driver = AsyncGraphDatabase.driver(
            self.uri,
            auth=(self.username, self.password),
            max_connection_pool_size=self.max_connection_pool_size,
            connection_acquisition_timeout=self.connection_acquisition_timeout,
            connection_timeout=self.connection_timeout,
            max_connection_lifetime=self.max_connection_lifetime,
            keep_alive=self.keep_alive,
        )
        self._driver = driver
        try:
            await driver.verify_connectivity(database=self.database)
        except BaseException:
            await driver.close()
            self._driver = None
            raise

    async def aclose(self) -> None:
        """Close the application-owned driver and all pooled Bolt connections."""

        driver, self._driver = self._driver, None
        if driver is not None:
            await driver.close()

    async def health(self) -> dict[str, Any]:
        """Verify connectivity without exposing credentials."""

        await self.driver.verify_connectivity(database=self.database)
        return {
            "status": "available",
            "backend": "neo4j",
            "database": self.database,
            "pool": self.connection_status(),
        }

    def connection_status(self) -> dict[str, Any]:
        return {
            "application_scoped": self._driver is not None,
            "pooled": self._driver is not None,
            "max_connection_pool_size": self.max_connection_pool_size,
            "connection_acquisition_timeout_seconds": (
                self.connection_acquisition_timeout
            ),
            "connection_timeout_seconds": self.connection_timeout,
            "max_connection_lifetime_seconds": self.max_connection_lifetime,
            "keep_alive": self.keep_alive,
        }
