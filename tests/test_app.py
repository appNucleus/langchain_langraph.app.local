from fastapi.testclient import TestClient

from app.main import app


def test_health() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"


def test_chat_echo() -> None:
    client = TestClient(app)
    response = client.post("/api/chat", json={"message": "hello"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["backend"] in {"echo", "ollama"}
    assert "response" in payload
