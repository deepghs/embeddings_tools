import glob
import json
import os
from typing import Optional

import autofaiss
import click
import numpy as np
from ditk import logging
from embedding_reader.get_file_list import get_file_list
from embedding_reader import EmbeddingReader
from hbutils.random import global_seed
from hbutils.system import TemporaryDirectory
from hfutils.operate import upload_directory_as_directory, get_hf_client
from tqdm import tqdm

from .utils import GLOBAL_CONTEXT_SETTINGS


@click.command(context_settings={**GLOBAL_CONTEXT_SETTINGS}, help='Index embedding data')
@click.option('-i', '--input_dir', 'input_dir', type=str, required=True,
              help='Input embeddings directory, should contain ids and embs subdirs.', show_default=True)
@click.option('-s', '--max_size', 'max_size', type=str, default='128MB',
              help='Max memory usage / file size of the index.', show_default=True)
@click.option('-r', '--repo_id', 'repo_id', type=str, required=True,
              help='Repository to upload to.', show_default=True)
@click.option('-n', '--index_name', 'index_name', type=str, default=None,
              help='Index name in repository.', show_default=True)
@click.option('-N', '--metrics_samples', 'metrics_samples', type=int, default=10000,
              help='Samples of metrics.', show_default=True)
@click.option('-s', '--seed', 'seed', type=int, default=0,
              help='Seed of metrics', show_default=True)
def cli(input_dir: str, max_size: str, repo_id: str, index_name: Optional[str],
        metrics_samples: int, seed: int):
    logging.try_init_root(level=logging.INFO)
    if not os.path.exists(input_dir):
        logging.error(f'Input directory {input_dir!r} not found, skipped.')
        return

    hf_client = get_hf_client()
    if not hf_client.repo_exists(repo_id=repo_id, repo_type='model'):
        hf_client.create_repo(repo_id=repo_id, repo_type='model')

    ids_dir = os.path.join(input_dir, 'ids')
    embs_dir = os.path.join(input_dir, 'embs')

    ids_filenames = sorted([os.path.relpath(file, ids_dir) for file in glob.glob(os.path.join(ids_dir, '*.npy'))])
    embs_filename = sorted([os.path.relpath(file, embs_dir) for file in glob.glob(os.path.join(embs_dir, '*.npy'))])
    assert ids_filenames == embs_filename, 'IDs files and embeddings files not match.'

    with TemporaryDirectory() as td:
        _, id_files = get_file_list(ids_dir, file_format='npy')
        ids = []
        for id_file in tqdm(id_files, desc='Concating ID files'):
            ids.append(np.load(id_file))
        ids = np.concatenate(ids)
        logging.info(f'Shape of IDs data: {ids.shape!r}, dtype: {ids.dtype!r}')
        all_ids_file = os.path.join(td, 'ids.npy')
        logging.info(f'Writing {all_ids_file!r} ...')
        np.save(all_ids_file, ids)

        if seed is not None:
            # native random, numpy, torch and faker's seeds are includes
            # if you need to register more library for seeding, see:
            # https://hansbug.github.io/hbutils/main/api_doc/random/state.html#register-random-source
            logging.info(f'Globally set the random seed {seed!r}.')
            global_seed(seed)

        idx = np.arange(ids.shape[0])
        np.random.shuffle(idx)
        idx = idx[:metrics_samples]

        logging.info('Picking samples from ')

        autofaiss.build_index(
            embeddings=embs_dir,
            index_path=os.path.join(td, 'knn.index'),
            index_infos_path=os.path.join(td, 'infos.json'),
            metric_type="ip",
            max_index_memory_usage=max_size,
        )

        with open(os.path.join(input_dir, 'meta.json'), 'r') as f:
            meta_info = json.load(f)

        index_name = index_name or f'{os.path.basename(input_dir)}_{meta_info["size"]}_{max_size}'
        upload_directory_as_directory(
            repo_id=repo_id,
            repo_type='model',
            local_directory=td,
            path_in_repo=index_name,
            message=f'Upload index {index_name!r} to {repo_id!r}',
        )


if __name__ == '__main__':
    cli()
