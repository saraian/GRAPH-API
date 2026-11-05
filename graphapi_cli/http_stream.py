"""Close camera relays cleanly when an upstream bridge shuts down."""
import http.client
import logging


def relay_response(source):
    with source:
        while True:
            try:
                chunk = source.read1(65536)
            except (http.client.HTTPException, OSError) as exc:
                logging.getLogger(__name__).info("Camera stream closed by bridge: %s", type(exc).__name__)
                return
            if not chunk:
                return
            yield chunk
