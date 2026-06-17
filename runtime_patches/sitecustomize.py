import asyncio


_original_get_event_loop = asyncio.get_event_loop


def _get_or_create_event_loop():
    try:
        return _original_get_event_loop()
    except RuntimeError as exc:
        if "There is no current event loop" not in str(exc):
            raise
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop


asyncio.get_event_loop = _get_or_create_event_loop
