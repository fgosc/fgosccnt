#!/usr/bin/env python3
import argparse
import logging
import csv
import cv2
from pathlib import Path

from fgosccnt import compute_hash_ce, compute_hash, imread, item_name, item_shortname

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

Item_dir = Path(__file__).resolve().parent / Path("item/")
category = ["point", "ce", "equip"]

def main(args):

    for c in category:
        search_dir = Item_dir / c
        files = search_dir.glob('**/*.png')
        for file in files:
            # name
            name = file.stem
            # id
            if c == "equip":
                try:
                    item_id = [k for k, v in item_shortname.items() if v == name][0]
                    name = item_name[item_id]
                except:
                    item_id = [k for k, v in item_name.items() if v == name][0]                    
            else:
                item_id = [k for k, v in item_name.items() if v == name][0]
            # hash
            img = imread(file)
            if c == "ce":
                item_hash = compute_hash_ce(img)
            else:
                item_hash = compute_hash(img)
            hash_hex = ""
            for h in item_hash[0]:
                hash_hex = hash_hex + "{:02x}".format(h)
            print("{},{},{}".format(item_id, name, hash_hex))
            
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--loglevel',
        choices=('DEBUG', 'INFO', 'WARNING'),
        default='WARNING',
        help='loglevel [default: WARNING]',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    logger.setLevel(args.loglevel)
    main(args)
