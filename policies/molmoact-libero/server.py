"""MolmoAct LIBERO checkpoint server — PLACEHOLDER.

Upstream openpi does not ship a `serve_policy_vllm.py`; this bundle's
server therefore can't be vendored verbatim like pi05-libero's can.
Replace the body of `main()` below with a real vLLM-backed implementation
that speaks the openpi websocket protocol — see
`openpi.serving.websocket_policy_server.WebsocketPolicyServer` for the
on-the-wire contract, and the pi05-libero bundle's `server.py` for the
shape of a working server.

Until then, this stub raises NotImplementedError at boot time so the
launcher fails fast with a clear message (the spawn never opens its port,
PolicyManager surfaces the early-exit, the user sees the message in stderr).
"""

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True,
                        help="HF or local checkpoint path (e.g., allenai/MolmoAct-7B-D-LIBERO-0812)")
    parser.add_argument("--port", type=int, required=True,
                        help="TCP port to bind the websocket server on")
    args = parser.parse_args()
    sys.stderr.write(
        "molmoact-libero/server.py is a placeholder — implement the vLLM-style "
        f"websocket server here (checkpoint={args.checkpoint!r}, port={args.port}). "
        "See the docstring in this file and the pi05-libero bundle's server.py "
        "for the websocket contract.\n"
    )
    raise NotImplementedError(
        "molmoact-libero/server.py is a stub. Add the real implementation "
        "(vLLM + openpi.serving.websocket_policy_server) before spawning."
    )


if __name__ == "__main__":
    main()
