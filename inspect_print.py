import pandas as pd
import textwrap
import pathlib
import json
import datetime

csv_path = pathlib.Path("data").glob("*.csv").__next__()
df = pd.read_csv(csv_path)

info = []
for col in df.columns:
    dtype = df[col].dtype
    nulls = df[col].isna().mean()
    unique = df[col].nunique() if dtype == "object" else ""
    info.append({"column": col,
                 "dtype": str(dtype),
                 "null_%": f"{nulls:.1%}",
                 "unique": unique})
print(pd.DataFrame(info))