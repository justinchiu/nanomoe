import os
from typing import Callable

import datasets
import tiktoken
import numpy as np
import altair as alt
import pandas as pd
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

enc = tiktoken.encoding_for_model("gpt-5")


def save_length_dist(
    dataset: datasets.Dataset,
    batch_fn: Callable[[datasets.Dataset], list[str]],
    filepath: str,
):
    batch_size = os.cpu_count()
    lengths = []
    with Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Tokenizing...", total=len(dataset))
        for start_idx in range(0, len(dataset), batch_size):
            examples = dataset.select(range(start_idx, min(start_idx+batch_size, len(dataset))))
            batch = batch_fn(examples)
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

    chart.save(filepath)


if __name__ == "__main__":
    os.makedirs("plots", exist_ok=True)
    math = datasets.load_dataset("zwhe99/DeepMath-103K")
    def batch_fn(examples):
        return [
            x
            for example in examples
            for x in [
                example["question"] + example["r1_solution_1"],
                example["question"] + example["r1_solution_2"],
                example["question"] + example["r1_solution_3"]
            ]
        ]
    #save_length_dist(math["train"], batch_fn, "plots/deepmath_length_distribution.pdf")

    ot = datasets.load_dataset("open-thoughts/OpenThoughts3-1.2M")
    def batch_fn(examples):
        return [x["conversations"][0]["value"] + x["conversations"][1]["value"] for x in examples]
    save_length_dist(ot["train"].shuffle().select(range(100_000)), batch_fn, "plots/ot_length_distribution.pdf")
