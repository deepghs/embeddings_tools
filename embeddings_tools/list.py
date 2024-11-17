import io
import json
import os
import re

import click
import numpy as np
import pandas as pd
from ditk import logging
from hbutils.scale import size_to_bytes_str
from hbutils.string import plural_word
from hbutils.system import TemporaryDirectory
from hfutils.operate import upload_directory_as_directory
from hfutils.utils import hf_fs_path
from huggingface_hub import HfFileSystem, HfApi
from huggingface_hub.constants import ENDPOINT
from huggingface_hub.hf_api import RepoFile
from tqdm.auto import tqdm

from .utils import GLOBAL_CONTEXT_SETTINGS, markdown_to_df


def _name_process(name: str):
    words = re.split(r'[\W_]+', name)
    return ' '.join([
        word.capitalize() if re.fullmatch('^[a-z0-9]+$', word) else word
        for word in words
    ])


_PERCENTAGE_METRICS = ('accuracy',)
HUGGINGFACE_CO_PAGE_TEMPLATE = ENDPOINT + "/{repo_id}/blob/{revision}/{filename}"


@click.command('list', context_settings={**GLOBAL_CONTEXT_SETTINGS},
               help='Publish model to huggingface model repository')
@click.option('--repository', '-r', 'repository', type=str, required=True,
              help='Repository for publishing model.', show_default=True)
@click.option('--revision', '-R', 'revision', type=str, default='main',
              help='Revision for pushing the model.', show_default=True)
def list_(repository: str, revision: str):
    logging.try_init_root(logging.INFO)
    hf_fs = HfFileSystem(token=os.environ.get('HF_TOKEN'))
    hf_client = HfApi(token=os.environ.get('HF_TOKEN'))

    names = [fn.split('/')[-2] for fn in hf_fs.glob(f'{repository}@{revision}/*/knn.index')]
    logging.info(f'{plural_word(len(names), "model")} detected in {repository}@{revision}')

    rows = []
    for name in tqdm(names):
        item = {'Name': name}
        infos = json.loads(hf_fs.read_text(f'{repository}@{revision}/{name}/infos.json'))

        if hf_fs.exists(f'{repository}@{revision}/{name}/metrics.json'):
            metrics = json.loads(hf_fs.read_text(f'{repository}@{revision}/{name}/metrics.json'))
            item["1-recall@20"] = f'{metrics["1-recall@20"] * 100:.1f}%'
            item["1-recall@40"] = f'{metrics["1-recall@40"] * 100:.1f}%'
            item["20-recall@20"] = f'{metrics["20-recall@20"] * 100:.1f}%'
            item["40-recall@40"] = f'{metrics["40-recall@40"] * 100:.1f}%'
        else:
            item["1-recall@20"] = 'N/A'
            item["1-recall@40"] = 'N/A'
            item["20-recall@20"] = 'N/A'
            item["40-recall@40"] = 'N/A'

        item['Size'] = size_to_bytes_str(infos['size in bytes'], sigfigs=3)
        item['AVG Speed (ms)'] = infos['avg_search_speed_ms']
        item['99% Speed (ms)'] = infos['99p_search_speed_ms']
        item['Reconstruction Error'] = f'{infos["reconstruction error %"]:.2f}%'

        item['Total'] = infos['nb vectors']
        item['Width'] = infos['vectors dimension']
        item['Compression Ratio'] = f'{infos["compression ratio"]:.4g}'

        item['Index Key'] = infos['index_key']
        for param_str in infos['index_param'].split(','):
            param_name, param_value = param_str.split('=', maxsplit=1)
            item[f'{param_name} (Param)'] = param_value

        repo_file: RepoFile = list(hf_client.get_paths_info(
            repo_id=repository,
            repo_type='model',
            paths=[f'{name}/knn.index'],
            expand=True,
        ))[0]
        last_commit_at = repo_file.last_commit.date.timestamp()

        item['created_at'] = last_commit_at
        rows.append(item)

    df = pd.DataFrame(rows)
    df = df.sort_values(by=['created_at'], ascending=[False])
    del df['created_at']
    df = df.replace(np.nan, 'N/A')

    with TemporaryDirectory() as td:
        with open(os.path.join(td, 'README.md'), 'w') as f:
            if not hf_fs.exists(hf_fs_path(
                    repo_id=repository,
                    repo_type='model',
                    filename='README.md',
                    revision=revision,
            )):
                print(df.to_markdown(index=False, numalign="center", stralign="center"), file=f)

            else:
                table_printed = False
                tb_lines = []
                with io.StringIO(hf_fs.read_text(hf_fs_path(
                        repo_id=repository,
                        repo_type='model',
                        filename='README.md',
                        revision=revision,
                )).rstrip() + os.linesep * 2) as ifx:
                    for line in ifx:
                        line = line.rstrip()
                        if line.startswith('|') and not table_printed:
                            tb_lines.append(line)
                        else:
                            if tb_lines:
                                df_c = markdown_to_df(os.linesep.join(tb_lines))
                                if 'Name' in df_c.columns and 'Index Key' in df_c.columns and \
                                        'Width' in df_c.columns:
                                    print(df.to_markdown(index=False, numalign="center", stralign="center"), file=f)
                                    table_printed = True
                                    tb_lines.clear()
                                else:
                                    print(os.linesep.join(tb_lines), file=f)
                            print(line, file=f)

                if not table_printed:
                    print(df.to_markdown(index=False, numalign="center", stralign="center"), file=f)

        upload_directory_as_directory(
            repo_id=repository,
            repo_type='model',
            revision=revision,
            path_in_repo='.',
            local_directory=td,
            message=f'Sync README for {repository}',
            hf_token=os.environ.get('HF_TOKEN'),
        )


if __name__ == '__main__':
    list_()
