import os
import time
import json
import asyncio
from typing import Dict, Optional

import aiohttp

from replay import producer
from advisor_agent import ElevatorAdvisor
from slack_sender import send_slack_alert
from tools.write_latest import write_latest

SCORE_URL = "http://127.0.0.1:8000/score"
RUN_MINUTES = 1

def risk_to_level(risk: float) -> str:
    if risk < 0.2:
        return "info"
    elif risk < 0.5:
        return "warning"
    else:
        return "critical"

async def post_score(session: aiohttp.ClientSession, row: Dict) -> Optional[Dict]:
    try:
        payload = {"features": {k: v for k, v in row.items() if not k.startswith("_")}}
        async with session.post(SCORE_URL, json=payload, timeout=10) as resp:
            if resp.status != 200:
                text = await resp.text()
                print(f"[score] HTTP {resp.status}: {text}")
                return None
            return await resp.json()
    except Exception as e:
        print(f"[score] error: {e}")
        return None

async def pipeline_loop():
    advisor = ElevatorAdvisor(model_name="gemini-1.5-flash", temperature=0.1)

    q: asyncio.Queue = asyncio.Queue(maxsize=1)
    prod_task = asyncio.create_task(producer(q, feature_names=None, limit=None))

    async with aiohttp.ClientSession() as session:
        start = time.time()
        idx = 0
        while True:
            if (time.time() - start) > RUN_MINUTES * 60:
                try:
                    await q.put(None)
                except Exception:
                    pass
                break
        
            item = await q.get()
            if item is None:
                break
        
            scored = await post_score(session, item)
            if not scored:
                continue

            risk = float(scored.get("risk", 0.0))
            top_features = scored.get("top_features", [])
            
            advice = advisor.advise(risk, top_features)

            level = risk_to_level(risk)
            msg = f"Risk {risk*100:.1f}% -> {advice}"
            send_slack_alert(msg, level=level)
            write_latest(id=ident, risk=risk, level=level, message=msg)

            ident = item.get("ID", "")
            print(f"[{idx}] ID={ident} risk={risk:.3f} level={level} | {advice}")
            idx += 1
    
    if not prod_task.done():
        prod_task.cancel()
        try:
            await prod_task
        except asyncio.CancelledError:
            pass

def main():
    asyncio.run(pipeline_loop())

if __name__ == "__main__":
    main()


                
