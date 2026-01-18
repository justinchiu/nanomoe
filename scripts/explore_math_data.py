import os
import datasets
import tiktoken
import numpy as np
import altair as alt
import pandas as pd
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

os.makedirs("plots", exist_ok=True)

math = datasets.load_dataset("zwhe99/DeepMath-103K")
enc = tiktoken.encoding_for_model("gpt-5")

train = math["train"]

batch_size = os.cpu_count()
lengths = []
with Progress(
    SpinnerColumn(),
    *Progress.get_default_columns(),
    TimeElapsedColumn(),
) as progress:
    task = progress.add_task("Tokenizing...", total=len(train))
    for start_idx in range(0, len(train), batch_size):
        examples = train.select(range(start_idx, min(start_idx+batch_size, len(train))))
        batch = [
            x
            for example in examples
            for x in [
                example["r1_solution_1"],
                example["r1_solution_2"],
                example["r1_solution_3"]
            ]
        ]
        encodings = enc.encode_batch(batch, num_threads=os.cpu_count())
        batch_lengths = [len(x) for x in encodings]
        lengths.extend(batch_lengths)
        progress.update(task, advance=len(examples))

# Create a dataframe for plotting
df = pd.DataFrame({"length": lengths})

# Create Altair histogram
chart = alt.Chart(df).mark_bar().encode(
    alt.X("length:Q", bin=alt.Bin(maxbins=50), title="Token Length"),
    alt.Y("count()", title="Frequency"),
).properties(
    title="Distribution of Solution Token Lengths",
    width=600,
    height=400,
)

chart.save("plots/length_distribution.pdf")
