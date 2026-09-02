import sys

from agent.maa_compat import AgentServer, Toolkit

# Imported for decorator registration.
from agent.actions import (
    configure_session,
    play_turn,
    record_event,
    recovery,
    route_conquest_tier,
    trace_runtime,
    warmup,
)
from agent.runtime import event_listener
from agent.runtime.task_cache import migrate_runtime_task_cache
from agent.recognitions import (
    card_selection,
    daily_task_reward,
    safe_entry,
    session_gate,
)


def _short_error(error: BaseException) -> str:
    return str(error).replace("\n", " ").replace("\r", " ").strip()[:160]


def main() -> None:
    """Start AgentServer with the socket id passed by MaaFramework."""
    started = False
    try:
        Toolkit.init_option("./")
        migrate_runtime_task_cache()
        if len(sys.argv) < 2:
            raise SystemExit("Usage: python -m agent.main <socket_id>")
        AgentServer.start_up(sys.argv[-1])
        started = True
        AgentServer.join()
    except BaseException as error:
        print(
            "[MarvelRuntimeIssue] event=agent_failed source=agent "
            f"reason={type(error).__name__} error={_short_error(error)}",
            flush=True,
        )
        raise
    finally:
        if started:
            AgentServer.shut_down()


if __name__ == "__main__":
    main()
