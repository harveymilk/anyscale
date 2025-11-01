import dotenv
dotenv.load_dotenv()

import pyarrow.parquet as pq
import pandas as pd
import s3fs

s3_path = "s3://anyscale-videos/results_parquet/54_000000_000000.parquet"
fs = s3fs.S3FileSystem(anon=False)

# Read Arrow table
table = pq.read_table(s3_path, filesystem=fs)

# Convert to pandas but keep Arrow-backed columns (no numpy dtype coercion)
df = table.to_pandas(types_mapper=pd.ArrowDtype)

print(df.dtypes)   # you'll see ArrowDtype(...) for some columns
print(df.head())
