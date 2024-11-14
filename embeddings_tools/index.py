import os

import click
import numpy as np
from ditk import logging
from embedding_reader.get_file_list import get_file_list
from tqdm import tqdm

from .utils import GLOBAL_CONTEXT_SETTINGS


@click.command(context_settings={**GLOBAL_CONTEXT_SETTINGS}, help='Index embedding data')
@click.option('-i', '--input_dir', 'input_dir', type=str, required=True,
              help='Input embeddings directory, should contain ids and embs subdirs.', show_default=True)
def cli(input_dir: str):
    logging.try_init_root(level=logging.INFO)
    if not os.path.exists(input_dir):
        logging.error(f'Input directory {input_dir!r} not found, skipped.')
        return

    ids_dir = os.path.join(input_dir, 'ids')
    embs_dir = os.path.join(input_dir, 'embs')

    _, id_files = get_file_list(ids_dir, file_format='npy')
    ids = []
    for id_file in tqdm(id_files, desc='Concating ID files'):
        ids.append(np.load(id_file))
    ids = np.concatenate(ids)
    logging.info(f'Shape of IDs data: {ids.shape!r}, dtype: {ids.dtype!r}')


if __name__ == '__main__':
    cli()
