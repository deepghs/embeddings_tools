import glob
import os

import click
import numpy as np
from ditk import logging
from hbutils.system import TemporaryDirectory
from hfutils.operate import get_hf_fs, download_archive_as_directory
from hfutils.utils import hf_normpath, hf_fs_path, parse_hf_fs_path
from tqdm import tqdm

from .utils import GLOBAL_CONTEXT_SETTINGS


@click.group(context_settings={**GLOBAL_CONTEXT_SETTINGS}, help='Collect embedding data')
def cli():
    pass  # pragma: no cover


@cli.command('hf', context_settings={**GLOBAL_CONTEXT_SETTINGS},
             help='Create embedding data from local npz directory, format: ids, embs')
@click.option('-o', '--output_dir', 'output_dir', type=str, required=True,
              help='Output directory, use a temp directory when not assigned.', show_default=True)
@click.option('-r', '--repo_id', 'repo_id', type=str, required=True,
              help='Repository to upload to.', show_default=True)
@click.option('-d', '--dir_in_repo', 'dir_in_repo', type=str, default='embs/SwinV2_v3',
              help='Directory in repository.', show_default=True)
def hf(output_dir: str, repo_id: str, dir_in_repo: str):
    logging.try_init_root(logging.INFO)
    hf_fs = get_hf_fs()

    files = [
        hf_normpath(os.path.relpath(parse_hf_fs_path(path).filename, dir_in_repo))
        for path in hf_fs.glob(hf_fs_path(
            repo_id=repo_id,
            repo_type='dataset',
            revision='main',
            filename=f'{dir_in_repo}/**/*.tar',
        ))
    ]

    os.makedirs(output_dir, exist_ok=True)
    for file in tqdm(files):
        body, _ = os.path.splitext(file.replace('/', '__'))
        dst_file = os.path.join(output_dir, f'{body}.npz')
        if os.path.exists(dst_file):
            logging.warning(f'{dst_file!r} already exist, skipped.')
            continue

        logging.info(f'Making {dst_file!r} with {hf_normpath(f"{dir_in_repo}/{file}")!r} ...')
        with TemporaryDirectory() as td:
            download_archive_as_directory(
                repo_id=repo_id,
                repo_type='dataset',
                file_in_repo=hf_normpath(f'{dir_in_repo}/{file}'),
                local_directory=td,
            )

            embs = []
            ids = []
            for npy_file in glob.glob(os.path.join(td, '*.npy'), recursive=True):
                id_ = int(os.path.splitext(os.path.basename(npy_file))[0])
                embedding = np.load(npy_file)
                assert len(embedding.shape) == 1
                ids.append(id_)
                embs.append(embedding)
            embs = np.stack(embs)
            assert len(embs.shape) == 2
            ids = np.array(ids)
            assert len(ids.shape) == 1
            assert ids.shape[0] == embs.shape[0]
            np.savez(dst_file, {
                'ids': ids,
                'embs': embs,
            })


if __name__ == '__main__':
    cli()
