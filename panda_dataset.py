
import os
from functools import partial
from modelscope import MsDataset

# Define mapping function
def download_url(example: dict, output_dir: str):
    output_dir = os.path.expanduser(output_dir)
    video_id: str = example['videoID']
    video_url: str = example['url']
    video_path: str = os.path.join(output_dir, f'{video_id}')
    if os.path.exists(video_path):
        print(f'** Reusing video {video_path}')
        example['video_path'] = video_path
        return example

    os.system(
        f"youtube-dl -o '{video_path}' {video_url}")
    example['video_path'] = video_path
    return example

download_url_func = partial(download_url, output_dir='output/dataset')

# Use sdk to load meta-data and download videos
ds = MsDataset.load('AI-ModelScope/panda-70m', subset_name='default', split='validation').to_hf_dataset()

ds = ds.map(download_url_func, num_proc=4)
print(next(iter(ds)))