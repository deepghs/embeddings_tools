import json
import os
import random
from tempfile import TemporaryDirectory
from typing import Optional

import click
import faiss
import numpy as np
from ditk import logging
from hbutils.string import plural_word
from hfutils.operate import get_hf_client
from hfutils.utils import hf_normpath
from huggingface_hub import CommitOperationAdd
from tqdm import tqdm

from .utils import GLOBAL_CONTEXT_SETTINGS


@click.group(context_settings={**GLOBAL_CONTEXT_SETTINGS}, help='Repack embedding data')
def cli():
    pass  # pragma: no cover


@cli.command('localx', context_settings={**GLOBAL_CONTEXT_SETTINGS},
             help='Repack embedding data from local npz directory, format: ids, embs, preds')
@click.option('-i', '--input_dir', 'input_dir', type=str, required=True,
              help='Input directory.', show_default=True)
@click.option('-o', '--output_dir', 'output_dir', type=str, default=None,
              help='Output directory, use a temp directory when not assigned.', show_default=True)
@click.option('-b', '--batch_size', 'batch_size', type=int, default=1000000,
              help='Batch size for repacking.', show_default=True)
@click.option('-p', '--prefix', 'prefix', type=str, default=None,
              help='Prefix for ids and files.', show_default=True)
@click.option('-L', '--level', 'level', type=int, default=4,
              help='Levels of pack id alignment.', show_default=True)
@click.option('-r', '--repo_id', 'repo_id', type=str, default=None,
              help='Repository to upload to.', show_default=True)
@click.option('-d', '--dir_in_repo', 'dir_in_repo', type=str, default=None,
              help='Directory in repository.', show_default=True)
@click.option('--raw', 'use_raw_embedding', is_flag=True, type=bool, default=False,
              help='Use raw embeddings, dont normalize them.', show_default=True)
def localx(input_dir: str, output_dir: str, batch_size: int, prefix: Optional[str] = None, level: int = 4,
           repo_id: Optional[str] = None, dir_in_repo: Optional[str] = None, use_raw_embedding: bool = False):
    logging.try_init_root(logging.INFO)
    if not os.path.exists(input_dir):
        logging.error(f'Input directory {input_dir!r} not found, skipped.')
        return

    with TemporaryDirectory() as td:
        output_dir = output_dir or td
        recorded_files = []
        ids, embs, all_ids, all_embs, samples, total_samples, current_ptr, width, sizes = \
            [], [], [], [], 0, 0, 0, None, []
        part_files = []

        def _save():

            nonlocal samples, ids, embs, current_ptr, width, sizes
            if samples:
                ids = np.concatenate(ids)
                embs = np.concatenate(embs)
                if not use_raw_embedding:
                    faiss.normalize_L2(embs)
                assert len(ids.shape) == 1
                assert len(embs.shape) == 2
                width = embs.shape[1]
                assert ids.shape[0] == embs.shape[0] == samples

                idx = np.arange(embs.shape[0])
                np.random.shuffle(idx)
                ids = ids[idx]
                embs = embs[idx]

                file_fmt = f'{{value:0{level}d}}.npy'
                if prefix is not None:
                    file_fmt = f'{prefix}_{file_fmt}'

                logging.info(f'Saving as part {current_ptr}, {plural_word(samples, "sample")} ...')
                dst_id_file = os.path.join(output_dir, 'ids', file_fmt.format(value=current_ptr))
                logging.info(f'Saving IDs to {dst_id_file!r} ...')
                os.makedirs(os.path.dirname(dst_id_file), exist_ok=True)
                np.save(dst_id_file, ids)
                recorded_files.append(dst_id_file)

                dst_embs_file = os.path.join(output_dir, 'embs', file_fmt.format(value=current_ptr))
                logging.info(f'Saving embeedings to {dst_embs_file!r} ...')
                os.makedirs(os.path.dirname(dst_embs_file), exist_ok=True)
                np.save(dst_embs_file, embs)
                recorded_files.append(dst_embs_file)
                part_files.append(file_fmt.format(value=current_ptr))

                sizes.append(ids.shape[0])
                current_ptr += 1
                ids = []
                embs = []
                samples = 0

        logging.info(f'Scanning for {input_dir!r} ...')
        tar_files = os.listdir(input_dir)
        random.shuffle(tar_files)
        for file in tqdm(tar_files):
            data = np.load(os.path.join(input_dir, file))
            ii, e = data['ids'], data['embs']
            if prefix is not None:
                ii = np.array([f'{prefix}_{x}' for x in ii])
            ids.append(ii)
            embs.append(e)
            all_ids.append(ii)
            all_embs.append(e)
            samples += e.shape[0]
            total_samples += e.shape[0]
            if samples >= batch_size:
                _save()

        _save()
        if not total_samples:
            logging.error(f'Nothing found for {input_dir!r}, skipped.')
            return

        all_ids = np.concatenate(all_ids)
        all_embs = np.concatenate(all_embs)
        assert len(all_ids.shape) == 1
        assert len(all_embs.shape) == 2
        assert all_ids.shape[0] == all_embs.shape[0] == total_samples

        meta_file = os.path.join(output_dir, 'meta.json')
        with open(meta_file, 'w') as f:
            json.dump({
                'size': total_samples,
                'embedding_width': width,
                'parts': current_ptr,
                'part_sizes': sizes,
                'part_files': part_files,
            }, f, indent=4, ensure_ascii=False, sort_keys=True)
        recorded_files.append(meta_file)

        if repo_id:
            hf_client = get_hf_client()
            hf_client.create_commit(
                repo_id=repo_id,
                repo_type='dataset',
                operations=[
                    CommitOperationAdd(
                        path_in_repo=hf_normpath(os.path.join(dir_in_repo or '.', os.path.relpath(file, output_dir))),
                        path_or_fileobj=file
                    ) for file in recorded_files
                ],
                commit_message=f'Upload {plural_word(total_samples, "embedding")} with ID(s)',
            )


if __name__ == '__main__':
    cli()
