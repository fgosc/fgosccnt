#!/usr/bin/env python3
# make  background.npz
import os

import cv2
import numpy as np
import urllib.request

from fgosccnt import img_hist, img_to_sig

aadb_url = "https://raw.githubusercontent.com/atlasacademy/aa-db/master/build/assets/list/"
img_dir = "data/misc/"
gold_flame = img_dir + "listframes3_bg.png"
silver_flame = img_dir + "listframes2_bg.png"
bronze_flame = img_dir + "listframes1_bg.png"
zero_flame = img_dir + "listframes0_bg.png"

file_gold = "data/misc/gold.png"
file_silver = "data/misc/silver.png"
file_bronze = "data/misc/bronze.png"
file_zero = "data/misc/zero.png"
output = "background.npz"

# 余白を落して拡大して保存
def download_file(url, filename):
    try:
        with urllib.request.urlopen(url + filename) as web_file:
            data = web_file.read()
            with open(img_dir  + filename, mode='wb') as local_file:
                local_file.write(data)
    except urllib.error.URLError as e:
        print(e)

def makeimg(file):
    img = cv2.imread(file)
    h, w = img.shape[:2]
    img = img[5: h-5, 5: w-5]

    #横幅188に拡大
    SIZE = 188
    img = cv2.resize(img, (0, 0),
            fx=SIZE/(w - 5), fy=SIZE/(w - 5),
            interpolation=cv2.INTER_AREA)

    return img

def main():
    for i in range(4):
        download_file(aadb_url, "listframes" + str(i) + "_bg.png")
    img_zero = makeimg(zero_flame)
    img_zero = img_zero[30:119, 7:25]
    target_hist = img_hist(img_zero)
    sig_zero = img_to_sig(target_hist)

    img_gold = makeimg(gold_flame)
    img_gold = img_gold[30:119, 7:25]
    target_hist = img_hist(img_gold)
    sig_gold = img_to_sig(target_hist)

    img_silver = makeimg(silver_flame)
    img_silver = img_silver[30:119, 7:25]
    target_hist = img_hist(img_silver)
    sig_silver = img_to_sig(target_hist)

    img_bronze = makeimg(bronze_flame)
    img_bronze = img_bronze[30:119, 7:25]
    target_hist = img_hist(img_bronze)
    sig_bronze = img_to_sig(target_hist)

    np.savez(output,
             sig_zero=sig_zero,
             sig_gold=sig_gold,
             sig_silver=sig_silver,
             sig_bronze=sig_bronze)

if __name__ == '__main__':
    os.makedirs(img_dir, exist_ok=True)
    main()
