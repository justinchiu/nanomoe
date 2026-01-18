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
    dataset: datasets.IterableDataset,
    batch_fn: Callable[[list[dict]], list[str]],
    filepath: str,
    max_examples: int,
):
    batch_size = os.cpu_count()
    lengths = []
    it = iter(dataset)
    with Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Tokenizing...", total=max_examples)
        for _ in range(0, max_examples, batch_size):
            examples = [next(it) for _ in range(batch_size)]
            batch = batch_fn(examples)
            encodings = enc.encode_batch(batch, num_threads=os.cpu_count())
            batch_lengths = [len(x) for x in encodings]
            lengths.extend(batch_lengths)
            progress.update(task, advance=len(examples))

    # Create a dataframe for plotting, capping at 32k+ bucket
    capped_lengths = [min(length, 32000) for length in lengths]
    df = pd.DataFrame({"length": capped_lengths})

    # Create Altair histogram with bins up to 32k, last bucket is 32k+
    chart = alt.Chart(df).mark_bar().encode(
        alt.X("length:Q", bin=alt.Bin(step=2000, extent=[0, 34000]), title="Token Length"),
        alt.Y("count()", title="Frequency"),
    ).properties(
        title="Distribution of Solution Token Lengths",
        width=600,
        height=400,
    )

    chart.save(filepath)


if __name__ == "__main__":
    os.makedirs("plots", exist_ok=True)
    math = datasets.load_dataset("zwhe99/DeepMath-103K", streaming=True)
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
    save_length_dist(math["train"].shuffle(), batch_fn, "plots/deepmath_length_distribution.pdf", 100_000)
    # CONCLUSION: CAN USE DEEPMATH!

    ot = datasets.load_dataset("open-thoughts/OpenThoughts3-1.2M", streaming=True)
    def batch_fn(examples):
        return [x["conversations"][0]["value"] + x["conversations"][1]["value"] for x in examples]
    save_length_dist(ot["train"].shuffle(), batch_fn, "plots/ot_length_distribution.pdf", 100_000)
    # CONCLUSION: CANT USE OPENTHOUGHTS

    nemomath = datasets.load_dataset("nvidia/Nemotron-CC-Math-v1", "4plus", streaming=True)
    def batch_fn(examples):
        return [example["text"] for example in examples]
    save_length_dist(nemomath["train"].shuffle(), batch_fn, "plots/nemomath_length_distribution.pdf", 100_000)
    # CONCLUSION: ?

