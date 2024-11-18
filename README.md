# embeddings_tools

Framework for processing embeddings and training indices with that.

## Installation

```shell
git clone https://github.com/deepghs/embeddings_tools.git
cd embeddings_tools
pip install -r requirements.txt
```

## Data Preparation

### Dataset Format

We use the following dataset format:

* Raw embeddings Dataset: A directory with many `.npz` files, each file include `ids` and `embs` keys.
  Shape of `ids`, `embs` is `(B, )` and `(B, dims)`. `B` is the samples count of this file, and `dims` is the dimensions
  of the embeddings. Each item in each file is strictly mapped one by one.
* Repacked embeddings Dataset : A directory with `.npy` files. Subdirectory `ids` contains npy files with
  shape `(B, )`, and the file in `embs` has the shape of `(B, dims)`. Like this

```text
SwinV2_v3_zerochan
|-- embs
|   |-- zerochan_0000.npy
|   |-- zerochan_0001.npy
|   |-- zerochan_0002.npy
|   `-- zerochan_0003.npy
|-- ids
|   |-- zerochan_0000.npy
|   |-- zerochan_0001.npy
|   |-- zerochan_0002.npy
|   `-- zerochan_0003.npy
`-- meta.json
```

Only the repacked embeddings can be used for indices training.

### Get A Raw Embeddings Dataset From HuggingFace Repository

```shell
python -m embeddings_tools.collect hf -r deepghs/danbooru2024-webp-4Mpixel -o /directory/of/raw_dataset
```

This will extract the `npy` embeddings files from the subdir `embs/SwinV2_v3` of the repository
[deepghs/danbooru2024-webp-4Mpixel](https://huggingface.co/datasets/deepghs/danbooru2024-webp-4Mpixel).
For large datasets this will take hours.

If the embeddings is not stored in this subdirectory, you can assign the subdir in repository like this

```shell
python -m embeddings_tools.collect -r deepghs/danbooru2024-webp-4Mpixel -o /directory/of/dataset -d embs/custom_one
```

### Repack Embeddings Dataset

```shell
python -m embeddings_tools.repack localx -i /directory/of/raw_dataset -o /directory/of/repacked_dataset
```

This will convert the raw dataset to a repacked embedding dataset.
And the embedding order in the raw datasets will be randomly shuffled to make the training result better.

If you need to add prefixes at the beginning of each embedding IDs, you can use `-p` option.
For examples, if you use `-p danbooru`, this will convert the original ID `114514` in the raw dataset to
`danbooru_114514` in the repacked dataset.

```shell
python -m embeddings_tools.repack localx -i /directory/of/raw_dataset -o /directory/of/repacked_dataset -p prefix
```

At some more advanced cases, we will have to repack and shuffle from multiple raw datasets,
you can use `-i prefix:/data/dir` to do this

```shell
python -m embeddings_tools.repack localx \
    -i danbooru:/data/raw/danbooru \
    -i zerochan:/data/raw/zerochan \
    -o /data/repacked/danbooru+zerochan
```

This command will create a repacked dataset at `/data/repacked/danbooru+zerochan`,
containing both danbooru and zerochan embeddings, with the IDs converted (e.g. ID `114514` in danbooru
will be `danbooru_114514`, and `1919810` in zerochan will be `zerochan_1919810`)

## Train And Publish the Index

```shell
python -m embeddings_tools.faiss -i /data/repacked/danbooru+zerochan -r my/repo_to_publish
```

After the training and evaluating has completed, all these data for the index will be uploaded to repository
`my/repo_to_publish` in a subdirectory.

The default max allowed inference memory size is `4GB`, enough for all the indices with <15M samples.
If you have to explicitly assign the inference size, you can use option like `-s 128MB` to do that.

For more options you can find them with `--help` option.

After you finished training and uploading, you can create a list in the README of that repository

```shell
python -m embeddings_tools.list -r my/repo_to_publish
```

## Use My Index

You can use your indices with the following script

```python
import json

import faiss
import numpy as np
from huggingface_hub import hf_hub_download

repo_id = 'my/repo_to_publish'
model_name = 'my_indices_name'

# load sample IDs, index and config of index
samples_ids = np.load(hf_hub_download(
    repo_id=repo_id,
    repo_type='model',
    filename=f'{model_name}/ids.npy',
))
knn_index = faiss.read_index(hf_hub_download(
    repo_id=repo_id,
    repo_type='model',
    filename=f'{model_name}/knn.index',
))
config = json.loads(open(hf_hub_download(
    repo_id=repo_id,
    repo_type='model',
    filename=f'{model_name}/infos.json',
)).read())["index_param"]
faiss.ParameterSpace().set_index_parameters(knn_index, config)

embeddings = ...  # embeddings with shape (1, dims)
faiss.normalize_L2(embeddings)

n_neighbours = 20  # find for 20 samples
dists, indexes = knn_index.search(embeddings, k=n_neighbours)
neighbours_ids = samples_ids[indexes][0]
for sample_id, dist in zip(neighbours_ids, dists[0]):  # print these 20 nearest samples
    print(f'Sample: {sample_id!r}, dist: {dist:.2f}')

```
