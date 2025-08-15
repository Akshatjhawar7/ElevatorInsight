# replay.py
# Emit one row every 10s via an asyncio.Queue.
# Works out-of-the-box with your `data/data_with_engineered_features.csv`.
# Optional: set SCORE_URL to POST each row to your FastAPI /score.

import os
import json
import time
import asyncio
from pathlib import Path
from typing import Optional, List, Dict

import pandas as pd

# ---------- Config ----------
DATA_PATH = Path("data/data_with_engineered_features.csv")
ROWS_PER_CHUNK = 5000                            # stream large files without loading all in memory
EMIT_INTERVAL_SEC = 10.0                         # <-- one row every 10 seconds

# ---------- Helpers ----------


async def producer(queue: asyncio.Queue, feature_names: Optional[List[str]] = None, limit: Optional[int] = None):
    """
    Read the CSV in chunks and push one row (dict) to the queue every EMIT_INTERVAL_SEC.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {DATA_PATH}")

    emitted = 0
    start_time = time.time()
    for chunk in pd.read_csv(DATA_PATH, chunksize=ROWS_PER_CHUNK):
        if feature_names is not None:
            keep = [c for c in feature_names if c in chunk.columns]           
            for id_col in ("ID", "elevator_id", "building_id"):
                if id_col in chunk.columns and id_col not in keep:
                    keep.append(id_col)
                    break
            chunk = chunk[keep]

        for _, row in chunk.iterrows():
            payload: Dict = row.to_dict()

            payload.pop("proxy_anomaly", None)

            payload["_emitted_at"] = time.time()

            await queue.put(payload)
            emitted += 1

            if limit and emitted >= limit:
                await queue.put(None)  # poison pill
                return

            await asyncio.sleep(EMIT_INTERVAL_SEC)

    await queue.put(None)  # no more data

async def _maybe_http_post(session, url: str, row: Dict) -> Optional[Dict]:
    try:
        async with session.post(url, json=row, timeout=10) as resp:
            return await resp.json()
    except Exception:
        return None

async def consumer(queue: asyncio.Queue):
    """
    Simple consumer:
    - If SCORE_URL is set and aiohttp is available, POST each row to /score and print the response.
    - Otherwise, just log a tiny summary to stdout.
    """
    session = None

    idx = 0
    try:
        while True:
            item = await queue.get()
            if item is None:
                break
            ident = item.get("ID", "")
            vib = item.get("vibration", item.get("vibration_mean_5m", ""))
            print(f"[{idx}] emit ID={ident} vibration={vib} at t={item['_emitted_at']:.0f}")

            idx += 1
    finally:
        if session is not None:
            await session.close()

async def main():
    q: asyncio.Queue = asyncio.Queue(maxsize=1)

    prod = asyncio.create_task(producer(q, feature_names=None, limit=None))
    cons = asyncio.create_task(consumer(q))

    try:
        await asyncio.gather(prod, cons)
    except asyncio.CancelledError:
        pass

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n Stopped by user")
