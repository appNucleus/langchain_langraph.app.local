# Caddy setup

Keep the FastAPI container bound to localhost through Docker Compose:

```env
HOST_BIND=127.0.0.1
APP_PORT=8001
```

Then add one of these to `/etc/caddy/Caddyfile`.

## Local/private domain with internal TLS

```caddyfile
langchain.home.arpa {
    tls internal
    reverse_proxy 127.0.0.1:8001
}
```

## Public domain

```caddyfile
ai.example.com {
    reverse_proxy 127.0.0.1:8001
}
```

Only expose ports `80` and `443` on the router/firewall. Do not expose Ollama, MCP, Redis, PostgreSQL, MongoDB, Neo4j, or MySQL directly to the internet.
