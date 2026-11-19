"""Check the LLM API the agent uses (.env AGENT_* / LLM_*). Prints what the server returns; never prints the key.
    py -3.12 -u scripts/agent_diag.py
"""
import os
import sys

import requests

sys.path.insert(0, ".")
from scripts.run_agent import model_settings

key, base, model = model_settings()
print(f"base {base} | model {model} | key set: {bool(key)}")
url = base.rstrip("/") + "/chat/completions"
msgs = [{"role": "user", "content": 'Return JSON: {"ok": true}'}]
for label, extra in (("json mode", {"response_format": {"type": "json_object"}}), ("plain", {}),
                     ("stream off", {"stream": False})):
    payload = {"model": model, "messages": msgs, "temperature": 0, "max_tokens": 50, **extra}
    try:
        r = requests.post(url, json=payload, headers={"Authorization": f"Bearer {key}"}, timeout=60)
        print(f"\n[{label}] HTTP {r.status_code} | content-type {r.headers.get('content-type')} | {len(r.text)} chars")
        print(r.text[:600])
    except Exception as e:
        print(f"\n[{label}] {type(e).__name__}: {e}")
