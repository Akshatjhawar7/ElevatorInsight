import json, os
from datetime import datetime, timezone

def write_latest(id: str, risk: float, level: str, message: str, out_path: str = 'docs/latest.json'):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    payload = {
        "id": id,
        "risk": float(risk),
        "level": level,
        "message": message,
        "at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00','Z')
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False)
