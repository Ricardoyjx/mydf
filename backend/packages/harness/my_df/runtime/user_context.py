from contextvars import ContextVar
from typing import Final, Protocol, runtime_checkable


@runtime_checkable
class CurrentUser(Protocol):
    """Structural type for the current authenticated user.

    Any object with an ``.id: str`` attribute satisfies this protocol.
    Concrete implementations live in ``app.gateway.auth.models.User``.
    """

    id: str


_current_user: Final[ContextVar[CurrentUser | None]] = ContextVar(
    "mf_df_current_user", default=None
)

DEFAULT_USER_ID: Final[str] = "default"


def get_effective_user_id() -> str:
    """Return the current user's id as a string, or DEFAULT_USER_ID if unset.

    Unlike :func:`require_current_user` this never raises — it is designed
    for filesystem-path resolution where a valid user bucket is always needed.
    """
    user = _current_user.get()
    if user is None:
        return DEFAULT_USER_ID
    return str(user.id)
