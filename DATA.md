# DATA.md – ElevatorInsight

**Dataset:** `elevator_requests.csv`  
**Source:** Kaggle – “Elevator Predictive Maintenance Dataset”  
**Rows (raw):** ≈ 1 000 000  **Period covered:** 2013-01-01 → 2015-12-31  
**Granularity:** One row per elevator door-cycle event (timestamp not provided in file)

---

## Column dictionary

| Column         | Type        | Missing % | Notes |
|----------------|------------:|----------:|-------|
| `ID`           | int64       | 0 % | Unique event index (monotonic) |
| `revolutions`  | float64     | 0 % | Door-motor revolutions per cycle |
| `humidity`     | float64     | 0 % | Shaft / cab relative humidity (%) |
| `vibration`    | float64     | 2.2 % | RMS vibration amplitude (mm/s) |
| `x1`           | float64     | 0 % | Proprietary sensor 1 (interpretation TBD) |
| `x2`           | float64     | 0 % | Proprietary sensor 2 |
| `x3`           | float64     | 0 % | Proprietary sensor 3 |
| `x4`           | float64     | 0 % | Proprietary sensor 4 |
| `x5`           | float64     | 0 % | Proprietary sensor 5 |

*(Update the “Notes” column as you learn what each `x*` channel measures.)*

---

## Proxy-label strategy

Because the file does **not** include an explicit failure flag, we generate a **proxy label** that marks rows showing extreme sensor behaviour likely to precede a door-roller fault.

```text
1. For each elevator ID:
   • Compute rolling z-scores (μ & σ over a 24-hour window) for
     `vibration` and `revolutions`.

2. Label rule:
   label = 1  if  (vibration_z > 3  OR  revolutions_z > 3)
               sustained for ≥ 10 consecutive rows
   else 0

Expected positive prevalence ≈ 2 %.
