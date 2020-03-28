#!/usr/bin/env python3
#-*- coding:utf-8 -*-
# FGO戦利品スクショの右上のドロップ数の数字を読む機械学習モデルを作成
# 一度にitemとchestのトレーニングデータを作るとメモリリークをしているのか
# おかしなデータができるので実行ファイル二つにわけている
#
# 以下のサイトを参考にした
# https://algorithm.joho.info/programming/python/hog-svm-classifier-py/

import cv2
import numpy as np
from pathlib import Path

item = 'item'             # training data directory

train = []
label = []

# Hog特徴の計算とラベリング
##def calc_hog(hog, pos_n, neg_n, win_size, cnt=1):
def calc_hog(hog, dirname, win_size):
    p_label_dir = Path('data') / Path(dirname) / Path('input')
    it_dir = p_label_dir.iterdir()

    # read image files from input directory
    for dir in it_dir:
        if dir.is_dir():
            p_filedir = Path(dir)
            it_file = p_filedir.glob('*.png')
        else:
            continue

        for file in it_file:
            file = str(file)
            img = cv2.imread(file, 0) #0はグレースケー
            img = cv2.resize(img, (win_size))
            train.append(hog.compute(img)) # 特徴量の格納
            label.append(int(dir.name))
    return np.array(train), np.array(label, dtype=int)


def main():
    # Hog特徴のパラメータ
    win_size = (120, 60)
    block_size = (16, 16)
    block_stride = (4, 4)
    cell_size = (4, 4)
    bins = 9
    pos_n = 3 # 正解画像の枚数
    neg_n = 3 # 不正解画像の枚数


    # Hog特徴の計算とラベリング
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, bins)
    train, label = calc_hog(hog, item, win_size)

    # Hog特徴からSVM識別器の作成
    svm = cv2.ml.SVM_create()
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setC(0.5)
    svm.train(train, cv2.ml.ROW_SAMPLE, label)
    svm.save(item + '.xml')

if __name__ == "__main__":
    main()
