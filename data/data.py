import os
import re
import json
import torch
import decord
import torchvision
import requests
import numpy as np
import boto3
from einops import rearrange
from typing import Dict, List, Tuple
import random
import traceback
import cv2

access_key = 'VVEGYBP0A4FFFPZDIUIC'
secret_key = 'XO3aoVDg4z2sJ8i8TDgDfnwReCJfdawLBgWPzwld'
# vimeo
endpoint_url='http://10.135.7.249:80'
# !!private date
s3_client = boto3.client(
's3',
endpoint_url=endpoint_url,
aws_access_key_id=access_key,
aws_secret_access_key=secret_key
)
s3_key = '000001_000050/1066683322.mp4'
s3_bucket = 'WebVid10M'
local_video_path = '/mnt/petrelfs/tianjie/projects/Pandora/data/12.mp4'

s3_client.download_file(s3_bucket, s3_key, local_video_path)
                        