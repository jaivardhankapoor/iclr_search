# %%
from openai import OpenAI
import json
import pandas as pd
from tqdm.auto import tqdm

client = OpenAI()


def get_embeddings_batch(texts, model="text-embedding-3-small"):
    response = client.embeddings.create(input=texts, model=model)
    return [data.embedding for data in response.data]


with open("/mnt/repos/iclr_search/iclr_2025_submissions.json", "r") as f:
    papers_dict = json.load(f)

# create df from papers_dict
df = pd.DataFrame(papers_dict)
df.head()

# %%
batch_size = 500
embeddings = []

tqdm.pandas(desc="Creating embeddings")

for i in tqdm(range(0, len(df), batch_size), desc="Processing batches", unit="batch"):
    batch_texts = (
        df.iloc[i : i + batch_size]
        .apply(
            lambda row: f"Title: {row['title']}\nAbstract: {row['abstract']} \nKeywords: {','.join(row['keywords'])}",
            axis=1,
        )
        .tolist()
    )
    batch_embeddings = get_embeddings_batch(batch_texts, model="text-embedding-3-small")
    embeddings.extend(batch_embeddings)

df["embedding"] = embeddings

# %%
df.to_csv(
    "/mnt/repos/iclr_search/iclr_2025_submissions_with_embeddings.csv", index=False
)


# %%
import pandas as pd

df = pd.read_csv("/mnt/repos/iclr_search/iclr_2025_submissions_with_embeddings.csv")
df.head()


# %%
import numpy as np

# convert embedding strings to numpy arrays
df["embedding"] = df["embedding"].apply(lambda x: np.array(eval(x)))

# stack the numpy arrays
embedding_array = np.stack(df["embedding"].values)

print(embedding_array.shape)

# %%
embedding_array[0]
# %%
# save the numpy array to a file
# np.save("/mnt/repos/iclr_search/embedding_array.npy", embedding_array)
np.save(
    "/mnt/repos/iclr_search/embedding_array_fp16.npy",
    embedding_array.astype(np.float16),
)

# %%
