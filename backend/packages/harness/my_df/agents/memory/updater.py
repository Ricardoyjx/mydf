from typing import Any

from my_df.agents.memory.storage import get_memory_storage


def get_memory_data(
    agent_name: str | None = None, *, user_id: str | None = None
) -> dict[str, Any]:
    """Get the current memory data via storage provider."""
    return get_memory_storage().load(agent_name, user_id=user_id)
