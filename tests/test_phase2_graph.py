import pytest
from app.graph import ChatAgent
from app.schemas.chat import ChatRequest
from app.settings import Settings

@pytest.mark.asyncio
async def test_echo_phase2_completes_without_database():
    agent=ChatAgent(Settings(llm_backend='echo'))
    response=await agent.ainvoke(ChatRequest(message='hello'))
    assert response.backend=='echo'
    assert response.metadata['phase']=='2'
    assert 'Message received' in response.response
