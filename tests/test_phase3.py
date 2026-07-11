import asyncio
import pytest

from app.schemas.execution import BudgetExceeded, ExecutionBudget
from app.state.in_memory import BoundedInMemoryStore
from app.tools.policies import SideEffectLevel, policy_for


def test_global_budget_enforces_model_limit():
    budget = ExecutionBudget(60, 1, 2, 2)
    budget.model_calls = 2
    with pytest.raises(BudgetExceeded):
        budget.check()


def test_mail_send_requires_confirmation():
    policy = policy_for('mail_send_draft')
    assert policy.confirmation_required is True
    assert policy.level == SideEffectLevel.EXTERNAL_COMMUNICATION


def test_memory_is_bounded():
    async def run():
        store = BoundedInMemoryStore(ttl_seconds=60, max_sessions=2, max_messages=2)
        await store.append('a', {'role':'user','content':'1'}, {'role':'assistant','content':'2'}, {'role':'user','content':'3'})
        assert len(await store.get('a')) == 2
        await store.append('b', {'role':'user','content':'b'})
        await store.append('c', {'role':'user','content':'c'})
        assert await store.get('a') == []
    asyncio.run(run())
