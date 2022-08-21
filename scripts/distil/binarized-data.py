# coding=utf-8
# Copyright 2019-present, the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocessing script before distillation.
"""
import argparse
import logging
import pickle
from multiprocessing import Pool

import numpy as np
import tqdm
from transformers import BertTokenizer


def binarize(sentence: str):
    token_ids = np.uint16(tokenizer.encode(sentence))
    return token_ids


tokenizer = BertTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(
    description="Preprocess the data to avoid re-doing it several times by (tokenization + token_to_ids)."
)
parser.add_argument(
    "--file_path", type=str, default="data/dump.txt", help="The path to the data."
)

parser.add_argument(
    "--dump_file", type=str, default="data/dump", help="The dump file prefix."
)
args = parser.parse_args()

logger.info(f"Loading text from {args.file_path}")
with open(args.file_path, "r", encoding="utf8") as fp:
    data = fp.readlines()

logger.info("Start encoding")
logger.info(f"{len(data)} examples to process.")

with Pool(12) as p:
    rslt = p.imap(binarize, data)
    rslt = tqdm.tqdm(rslt, total=len(data), mininterval=5)
    rslt = list(rslt)

logger.info("Finished binarization")
logger.info(f"{len(data)} examples processed.")

dp_file = f"{args.dump_file}.pickle"
logger.info(f"Dump to {dp_file}")
with open(dp_file, "wb") as handle:
    pickle.dump(rslt, handle, protocol=pickle.HIGHEST_PROTOCOL)
