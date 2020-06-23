#!/usr/bin/env python3
import sys
import re
import argparse
import cv2
import pageinfo
import numpy as np
from pathlib import Path
from collections import Counter
import csv

progname = "FGOスクショカウント"
version = "0.3.0"

Item_dir = Path(__file__).resolve().parent / Path("item/")
train_item = Path(__file__).resolve().parent / Path("item.xml") #アイテム下部
train_chest = Path(__file__).resolve().parent / Path("chest.xml") #ドロップ数
train_card = Path(__file__).resolve().parent / Path("card.xml") #ドロップ数

hasher = cv2.img_hash.PHash_create()

#恒常アイテムのハッシュ値
dist_item ={
    'QP':np.array([[ 82, 9, 116, 145, 236, 2, 32, 3]], dtype='uint8'),
    '爪':np.array([[254,   7,  81, 188,  13, 198, 115,  18]], dtype='uint8'),
    '心臓':np.array([[ 94, 131,  49, 137,  73,  76,   0,  90]], dtype='uint8'),
    '逆鱗':np.array([[142, 121,  57, 147, 103,   7,  78,  11]], dtype='uint8'),
    '根':np.array([[ 86,  41, 169,  73, 164,  22,  41,   9]], dtype='uint8'),
    '幼角':np.array([[146, 41, 86, 41, 214, 73, 165, 210]], dtype='uint8'),
    '涙石':np.array([[22, 33, 88, 166, 104, 19, 153, 76]], dtype='uint8'),
    '脂':np.array([[ 94,   5, 161,  97,  27,  88,  20, 132]], dtype='uint8'),
    'ランプ':np.array([[182,  41, 197, 114,  21, 204, 131, 120]], dtype='uint8'),
    'スカラベ':np.array([[190,  13,  67, 113,  24, 128, 132,   0]], dtype='uint8'),
    '産毛':np.array([[ 30,  51, 217,  41, 164,  22, 154,  74]], dtype='uint8'),
    '胆石':np.array([[ 22, 129,  97, 162, 156,  99,  38, 148]], dtype='uint8'),
    '神酒':np.array([[ 62, 132,  37, 169,  89,  89, 100,  78]], dtype='uint8'),
    '炉心':np.array([[122,   5,  36, 171,  25,  88,  17, 136]], dtype='uint8'),
    '鏡':np.array([[254,   3, 196,  86, 163, 165,  41,  41]], dtype='uint8'),
    '卵':np.array([[ 22, 161, 237,  72,  25,   9,  73,  72]], dtype='uint8'),
    'カケラ':np.array([[ 26, 129, 102,  88,   5, 154,   5,  44]], dtype='uint8'),
    '実':np.array([[126, 129,  82, 101, 140,  10, 120,  30]], dtype='uint8'),
    '種':np.array([[ 62, 133, 169, 41, 90, 148, 134, 32]], dtype='uint8'),
    'ランタン':np.array([[166, 201, 89, 154, 166, 100, 121, 38]], dtype='uint8'),
    '八連':np.array([[126, 5, 165, 201,  25, 150, 98, 36]], dtype='uint8'),
    '蛇玉':np.array([[86, 165, 195, 114, 185, 177, 137, 114]], dtype='uint8'),
    '羽根':np.array([[166, 187,  77,  68, 105,  20,  68, 162]], dtype='uint8'),
    '歯車':np.array([[ 94,  18, 225, 121,  25, 141,  12, 228]], dtype='uint8'),
    '頁':np.array([[223,   9, 246,  28, 178, 236,  91, 168]], dtype='uint8'),
    'ホム':np.array([[ 86, 225, 106, 132, 177,  25, 173, 101]], dtype='uint8'),
    '蹄鉄':np.array([[120, 133, 115, 185,  24, 196, 100,  34]], dtype='uint8'),
    '勲章':np.array([[150,  82, 109, 173, 181,  20, 108,  43]], dtype='uint8'),
    '貝殻':np.array([[26, 165, 213, 72, 140, 214, 176, 73]], dtype='uint8'),
    '勾玉':np.array([[254,   5,  98, 233,  92, 182,  13, 204]], dtype='uint8'),
    '結氷':np.array([[126, 129, 108, 198, 147, 106, 201,  54]], dtype='uint8'),
    '指輪':np.array([[122,   7, 197, 177, 123,  11,  81,  24]], dtype='uint8'),
    'オーロラ':np.array([[ 94, 163,  85,  20, 169, 137,  36, 105]], dtype='uint8'),
    '鈴':np.array([[122,   5, 193, 131,  60,  42,  82,  22]], dtype='uint8'),
    '矢尻':np.array([[190,   1, 104, 168,  77,  66, 180, 130]], dtype='uint8'),
    '冠':np.array([[233,  29, 234,  65,  62,  69, 250, 233]], dtype='uint8'),
    '霊子':np.array([[250,   6,  61, 185, 139,  82,  86, 212]], dtype='uint8'),
    '証':np.array([[94, 5, 161, 88, 6, 70, 33, 25]], dtype='uint8'),
    '骨':np.array([[82, 75, 37, 149,  85,  33, 168, 165]], dtype='uint8'),
    '牙':np.array([[58, 131,  21, 217, 237, 101,  44, 176]], dtype='uint8'),
    '塵':np.array([[222, 1, 104, 26, 134, 164, 42, 17]], dtype='uint8'),
    '鎖':np.array([[ 14,  83, 177,  25, 204, 169,  38,  22]], dtype='uint8'),
    '毒針':np.array([[ 90, 181,  41,  75, 211, 178,  52, 108]], dtype='uint8'),
    '髄液':np.array([[ 38,  25, 114, 205, 154,   7,   2, 100]], dtype='uint8'),
    '鉄杭':np.array([[ 76, 141,  51,  99, 227,  99,  48,  48]], dtype='uint8'),
    '火薬':np.array([[110, 147,  57, 157,  69, 53, 194, 42]], dtype='uint8'),
    '剣秘':np.array([[106, 150, 230,  97,  31, 216, 153, 174]], dtype='uint8'),
    '弓秘':np.array([[104,  22, 102, 227,  63, 216, 205, 234]], dtype='uint8'),
    '槍秘':np.array([[232,   6,  50,  97, 159, 176, 207, 154]], dtype='uint8'),
    '騎秘':np.array([[42, 142, 118,  97,  31, 153, 207,  58]], dtype='uint8'),
    '術秘':np.array([[122,   6,  70,  51,  31, 232, 153, 110]], dtype='uint8'),
    '殺秘':np.array([[106, 158, 102, 227,  27, 224, 143, 154]], dtype='uint8'),
    '狂秘':np.array([[234,  30, 102, 194,  27, 200,  15, 154]], dtype='uint8'),
    '剣魔':np.array([[94, 161, 41, 137, 9, 72,76, 66]], dtype='uint8'),
    '弓魔':np.array([[ 22, 225,  41,  41, 136,  72,  92,  66]], dtype='uint8'),
    '槍魔':np.array([[ 86, 161, 169,  57,   9,  72,  24,  66]], dtype='uint8'),
    '騎魔':np.array([[94, 165, 41, 25, 9, 76, 88, 198]], dtype='uint8'),
    '術魔':np.array([[ 94, 161,  49,  41,   9,   8, 100,  66]], dtype='uint8'),
    '殺魔':np.array([[ 94, 165, 185,  41,  12,  72,  92,  66]], dtype='uint8'),
    '狂魔':np.array([[94, 165, 185, 41, 12, 72, 88, 102]], dtype='uint8'),
    '剣輝':np.array([[ 30, 225,  41,  90,  82,   6, 166,  41]], dtype='uint8'),
    '弓輝':np.array([[30, 225, 169, 121, 214, 38,134, 36]], dtype='uint8'),
    '槍輝':np.array([[ 30, 225, 169,  27,  70, 198, 166,  33]], dtype='uint8'),
    '騎輝':np.array([[30, 225, 169, 89, 86, 6, 128, 164]], dtype='uint8'),
    '術輝':np.array([[ 30, 225, 169,  73,  86, 150, 166,  41]], dtype='uint8'),
    '殺輝':np.array([[30, 229, 169, 89, 70, 22, 166, 33]], dtype='uint8'),
    '狂輝':np.array([[ 30, 229, 169, 121,  86, 150, 132,  36]], dtype='uint8'),
    '剣モ':np.array([[150, 161,  89,  73, 100, 155, 166,  38]], dtype='uint8'),
    '弓モ':np.array([[ 70, 153,  35,  66, 133,  27,  61,  58]], dtype='uint8'),
    '槍モ':np.array([[214, 169,  50,  73, 164,  13, 102, 146]], dtype='uint8'),
    '騎モ':np.array([[54, 233,  25, 158, 101,  58, 137, 68]], dtype='uint8'),
    '術モ':np.array([[ 70, 161,  24, 183, 100,  83, 156,  98]], dtype='uint8'),
    '殺モ':np.array([[102, 185, 204, 210,  37,  38,  17,  78]], dtype='uint8'),
    '狂モ':np.array([[ 14,  73, 163, 211,  73, 134, 100,  43]], dtype='uint8'),
    '剣ピ':np.array([[150, 177, 73, 73, 100, 154, 166, 36]], dtype='uint8'),
    '弓ピ':np.array([[ 86, 153,  99,  66, 132,  89,  61,  56]], dtype='uint8'),
    '槍ピ':np.array([[214, 169,  58, 216, 164,  44, 102, 146]], dtype='uint8'),
    '騎ピ':np.array([[54, 233,  25, 154, 101,  58, 137, 100]], dtype='uint8'),
    '術ピ':np.array([[ 70, 233,  24, 178, 108,  83, 172,  98]], dtype='uint8'),
    '殺ピ':np.array([[102, 185, 204, 210,  53,  38, 153,  78]], dtype='uint8'),
    '狂ピ':np.array([[14, 105, 163,  82,  89, 150, 116, 107]], dtype='uint8'),
}

#秘石を見分けるハッシュ値
dist_hiseki = {
    '剣秘':np.array([[101, 225,  88, 190, 105, 145,  88, 225]], dtype='uint8'),
    '弓秘':np.array([[ 29, 240, 102, 190, 122, 197,  25, 240]], dtype='uint8'),
    '槍秘':np.array([[121, 252,  88, 243,  38,  79, 148, 174]], dtype='uint8'),
    '騎秘':np.array([[ 70,  49,  73, 234, 121, 156,  24, 196]], dtype='uint8'),
    '術秘':np.array([[107, 102,  16, 157,  86, 113,  13, 192]], dtype='uint8'),
    '殺秘':np.array([[ 95, 121,  73, 238,  90, 165,  20,  80]], dtype='uint8'),
    '狂秘':np.array([[126, 153,  89, 230, 137, 165,  16, 112]], dtype='uint8'),
}

#魔石を見分けるハッシュ値
dist_maseki = {
    '剣魔':np.array([[153, 158, 230, 103, 126, 111, 175, 92]], dtype='uint8'),
    '弓魔':np.array([[227, 242, 24, 254, 141, 255, 230, 189]], dtype='uint8'),
    '槍魔':np.array([[131, 2, 60, 252, 248, 181, 106, 209]], dtype='uint8'),
    '騎魔':np.array([[185, 202, 118, 52, 206, 227, 247, 56]], dtype='uint8'),
    '術魔':np.array([[145, 153, 110, 98, 120, 158, 242, 59]], dtype='uint8'),
    '殺魔':np.array([[233, 230, 94, 27, 239, 251, 235, 165]], dtype='uint8'),
    '狂魔':np.array([[1, 230, 238, 154, 126, 91, 175, 167]], dtype='uint8'),
}

#輝石を見分けるハッシュ値
dist_kiseki = {
    '剣輝':np.array([[152, 191, 158, 101,  30, 102, 165,  62]], dtype='uint8'),
    '弓輝':np.array([[ 96, 171, 154,  86, 143,  18, 230, 185]], dtype='uint8'),
    '槍輝':np.array([[  0,  11, 126,  92, 218, 180,  34,  89]], dtype='uint8'),
    '騎輝':np.array([[ 56, 203, 118,  20, 142, 163, 103,  56]], dtype='uint8'),
    '術輝':np.array([[ 16, 153,  78,  98,  58, 150, 242,  59]], dtype='uint8'),
    '殺輝':np.array([[224, 175,  94,  81, 175,  90,  97, 166]], dtype='uint8'),
    '狂輝':np.array([[ 17, 239, 254,  90, 126,  91, 166, 175]], dtype='uint8'),
}

#種火を見分けるハッシュ値
dist_tanebi = {
    '全種火':np.array([[241,  88, 142, 178,  78, 205, 238,  43]], dtype='uint8'),
    '剣種火':np.array([[241,  42,  46, 187,  79, 253, 110, 172]], dtype='uint8'),
    '弓種火':np.array([[241, 248, 174, 186,  79, 253,  78, 172]], dtype='uint8'),
    '槍種火':np.array([[241, 248, 174, 186,  15, 253,  79, 172]], dtype='uint8'),
    '騎種火':np.array([[241,  43,  46, 187,  79, 249, 238, 172]], dtype='uint8'),
    '術種火':np.array([[241,  40,  46, 187,  15, 253, 110, 172]], dtype='uint8'),
    '殺種火':np.array([[241,  42,  46, 186, 111, 249, 238, 172]], dtype='uint8'),
    '狂種火':np.array([[241,  43,  46, 187, 111, 249, 238, 168]], dtype='uint8'),
    '剣灯火':np.array([[243, 222,  46,  99, 207, 153, 126, 237]], dtype='uint8'),
    '弓灯火':np.array([[243, 220,  46, 227, 207, 153, 126, 205]], dtype='uint8'),
    '槍灯火':np.array([[241, 220,  46,  99, 207,  27, 126, 205]], dtype='uint8'),
    '騎灯火':np.array([[115,  94,  46,  99, 207, 153, 254, 237]], dtype='uint8'),
    '術灯火':np.array([[243, 222,  46,  99, 207, 153, 126, 237]], dtype='uint8'),
    '殺灯火':np.array([[243, 222,  46, 227, 207, 185, 254, 237]], dtype='uint8'),
    '狂灯火':np.array([[123,  94,  46,  99, 207, 185, 246, 237]], dtype='uint8'),
    '剣大火':np.array([[ 51, 120,  78, 147, 182, 104,  43, 230]], dtype='uint8'),
    '弓大火':np.array([[243, 248,  74, 147, 182, 106,  43, 166]], dtype='uint8'),
    '槍大火':np.array([[243, 248,  74, 147, 182, 110,  47, 166]], dtype='uint8'),
    '騎大火':np.array([[ 51, 120,  14, 147, 182, 104,  43, 230]], dtype='uint8'),
    '術大火':np.array([[ 51, 120,  74, 147, 182, 110,  43, 230]], dtype='uint8'),
    '殺大火':np.array([[ 51, 120,  78, 147, 182, 104,  43, 230]], dtype='uint8'),
    '狂大火':np.array([[ 51, 120,  14, 147, 182, 232,  41, 230]], dtype='uint8'),
    '剣猛火':np.array([[ 11,  40, 244, 186, 158, 243, 207, 163]], dtype='uint8'),
    '弓猛火':np.array([[ 11,  40, 244, 186, 158, 211, 207, 163]], dtype='uint8'),
    '槍猛火':np.array([[ 11,  40, 244, 186, 158, 211, 207, 163]], dtype='uint8'),
    '騎猛火':np.array([[ 11,  42, 252, 186, 158, 243, 207, 161]], dtype='uint8'),
    '術猛火':np.array([[ 11,  40, 244, 186, 158, 243, 207, 163]], dtype='uint8'),
    '殺猛火':np.array([[ 11,  40, 244, 186, 190, 243, 207, 161]], dtype='uint8'),
    '狂猛火':np.array([[ 11,  42, 252, 186, 158, 243, 207, 160]], dtype='uint8'),
    '剣業火':np.array([[ 41,  47, 254, 248,  47,  94, 123, 175]], dtype='uint8'),
    '弓業火':np.array([[ 41, 175, 254, 248,  47,  94, 123, 175]], dtype='uint8'),
    '槍業火':np.array([[ 41,  47, 254, 248,  47,  94, 123, 175]], dtype='uint8'),
##    '騎業火':np.array(, dtype='uint8'),
    '術業火':np.array([[ 41,  47, 254, 248,  47,  94, 123, 175,]], dtype='uint8'),
    '殺業火':np.array([[ 41,  47, 238, 248,  47, 222, 123, 175,]], dtype='uint8'),
    '狂業火':np.array([[ 41,  47, 190, 248,  47, 222, 115, 160,]], dtype='uint8'),
    '全種火変換':np.array([[ 75, 248, 248,   7, 244, 172,   6, 182]], dtype='uint8'),
    '剣種火変換':np.array([[ 11, 248, 248,  15, 244, 172,   6, 150]], dtype='uint8'),
    '弓種火変換':np.array([[ 75, 248, 248,  15, 244, 172,   6, 150]], dtype='uint8'),
    '槍種火変換':np.array([[ 75, 248, 248,  15, 244, 172,   6, 150]], dtype='uint8'),


    '騎種火変換':np.array([[ 11, 248, 248,  15, 244, 172,   6, 150]], dtype='uint8'),
    '術種火変換':np.array([[ 75, 248, 248,  15, 244, 172,   6, 150]], dtype='uint8'),
    '殺種火変換':np.array([[ 11, 248, 249,  15, 244, 172,   6, 150]], dtype='uint8'),
    '狂種火変換':np.array([[ 11, 248, 249,  15, 244, 172,   6, 150]], dtype='uint8'),
    '剣灯火変換':np.array([[ 11, 248, 248,   7, 244, 172,   6, 134]], dtype='uint8'),
    '弓灯火変換':np.array([[139, 248, 248,   7, 244, 172,   6, 134]], dtype='uint8'),
    '槍灯火変換':np.array([[139, 248, 248,   7, 116, 172,   6, 134]], dtype='uint8'),


    '騎灯火変換':np.array([[ 11, 248, 248,   7, 244, 172,   6, 134]], dtype='uint8'),
    '術灯火変換':np.array([[139, 248, 249,   7, 116, 172,   6, 134]], dtype='uint8'),
    '殺灯火変換':np.array([[139, 248, 249,   7, 244, 172,   6, 134]], dtype='uint8'),
    '狂灯火変換':np.array([[ 11, 248, 249,   7, 244, 172,   6, 134]], dtype='uint8'),
    '剣大火変換':np.array([[  7, 249, 248, 191, 166, 143,  46, 244]], dtype='uint8'),
    '弓大火変換':np.array([[  7, 249, 248, 191, 166, 143,  46, 244]], dtype='uint8'),
    '槍大火変換':np.array([[  3, 249, 248, 191, 166, 143,  46, 244]], dtype='uint8'),
    '騎大火変換':np.array([[  7, 249, 248, 191, 166, 143,  46, 244]], dtype='uint8'),
    '術大火変換':np.array([[  3, 249, 248, 191, 166, 143,  46, 244]], dtype='uint8'),
    '殺大火変換':np.array([[  3, 249, 248, 191, 166, 143,  46, 244]], dtype='uint8'),
    '狂大火変換':np.array([[  7, 249, 248, 191, 166, 143, 174, 244]], dtype='uint8'),
    '剣猛火変換':np.array([[ 11, 248, 248, 159,  38, 143, 206, 240]], dtype='uint8'),
    '弓猛火変換':np.array([[ 11, 248, 248, 159,  38, 143, 206, 240]], dtype='uint8'),
    '槍猛火変換':np.array([[ 11, 248, 248, 159,  38, 143, 206, 240]], dtype='uint8'),
    '騎猛火変換':np.array([[ 11, 248, 248, 159,  38, 135, 206, 240]], dtype='uint8'),
    '術猛火変換':np.array([[ 11, 248, 248, 159,  38, 143, 206, 240]], dtype='uint8'),
    '殺猛火変換':np.array([[ 11, 248, 248, 159,  38, 143, 206, 240]], dtype='uint8'),
    '狂猛火変換':np.array([[ 11, 122, 248, 159,  38, 135, 206, 240]], dtype='uint8'),
}

#種火のレアリティを見分けるハッシュ値
dist_tanebi_rarity = {
    '種火':np.array([[161,  37, 182, 205, 217, 217,  89, 190]], dtype='uint8'),
    '灯火':np.array([[233, 108,  54, 198, 219, 123, 189, 181]], dtype='uint8'),
    '大火':np.array([[211, 180,  61,   6, 188, 171,  45, 231]], dtype='uint8'),
    '猛火':np.array([[157,  51, 164, 180, 114,  92,  88, 161]], dtype='uint8'),
    '業火':np.array([[225, 127, 240,  27,  46, 218,  24,   2]], dtype='uint8'),
    '種火変換':np.array([[ 15, 224, 159, 112, 217,  31, 224, 159]], dtype='uint8'),
    '灯火変換':np.array([[143, 224,  63,  68, 216,  63, 240, 159]], dtype='uint8'),
    '大火変換':np.array([[ 30,  30,  32,  96, 136,  63, 240, 244]], dtype='uint8'),
    '猛火変換':np.array([[158,  30, 164,  64, 160,  95, 240, 122]], dtype='uint8'),
}
#種火のクラス見分けるハッシュ値
dist_tanebi_class = {
    '全種火':np.array([[161, 223,  56,  15,  62,  41, 199,   5]], dtype='uint8'),
    '剣種火':np.array([[217,  95, 229,  49,  62,  39,  30,  10]], dtype='uint8'),
    '弓種火':np.array([[115,  63, 137,  53, 173,  42, 106,   5]], dtype='uint8'),
    '槍種火':np.array([[227,  51,  37,  44, 120, 178,  43,  65]], dtype='uint8'),
    '騎種火':np.array([[153, 111,  22,  22, 230,  35,  63,   8]], dtype='uint8'),
    '術種火':np.array([[213, 157,  39,  35,  60,  86,  58, 141]], dtype='uint8'),
    '殺種火':np.array([[249,  54,  22,  57, 177,  41,  39,  74]], dtype='uint8'),
    '狂種火':np.array([[ 73, 167, 246,  30,  58,  25, 247,  33]], dtype='uint8'),
    '剣灯火':np.array([[217, 223, 167,  33,  62,  35,  30,  10]], dtype='uint8'),
    '弓灯火':np.array([[119, 191, 137,  37, 141,  42, 106,   5]], dtype='uint8'),
    '槍灯火':np.array([[227, 243,   5,  44, 120, 178,  43,  73]], dtype='uint8'),
    '騎灯火':np.array([[153, 111,  22,  22, 230,  35,  59,   8]], dtype='uint8'),
    '術灯火':np.array([[221, 157,  39,  35,  60,  86,  58,  13]], dtype='uint8'),
    '殺灯火':np.array([[249,  54,  22,  57, 176,  41,  39,  74]], dtype='uint8'),
    '狂灯火':np.array([[ 73, 167, 246,  30,  42,  25, 247,  33]], dtype='uint8'),
    '剣大火':np.array([[217,  91, 229,  33,  58,  35,  62,  10]], dtype='uint8'),
    '弓大火':np.array([[115,  45, 137,  53, 173,  42, 106,   5]], dtype='uint8'),
    '槍大火':np.array([[227,  51,  37,  44, 120, 178,  43,  72]], dtype='uint8'),
    '騎大火':np.array([[185, 105,  22,  60, 230,  35,  63,   8]], dtype='uint8'),
    '術大火':np.array([[149, 141,  38,  51, 188,  86,  58, 141]], dtype='uint8'),
    '殺大火':np.array([[249,  38,  22,  57, 248,  41,  55,  74]], dtype='uint8'),
    '狂大火':np.array([[ 73, 167, 246,  94,  58,  27, 255,  33]], dtype='uint8'),
    '剣猛火':np.array([[217,  95, 245,  49,  62,  47,  30,  10]], dtype='uint8'),
    '弓猛火':np.array([[113,  55, 152,  55, 125,  43, 106,   7]], dtype='uint8'),
    '槍猛火':np.array([[225,  35,  20,  44, 120,  62,  59,  72]], dtype='uint8'),
    '騎猛火':np.array([[217, 109,  22,  54, 238,  43,  63,   8]], dtype='uint8'),
    '術猛火':np.array([[ 81, 221,  38,  35,  60,  86,  26, 141]], dtype='uint8'),
    '殺猛火':np.array([[249,  39,  86,  57, 244,  43,  39,  74]], dtype='uint8'),
    '狂猛火':np.array([[ 89,  39, 246,  94,  58,  27, 119,  33]], dtype='uint8'),
    '剣業火':np.array([[217, 223, 161,  57,  62,  43,  30,  10]], dtype='uint8'),
    '弓業火':np.array([[115, 191, 136,  63,  45,  42,  42,   5]], dtype='uint8'),
    '槍業火':np.array([[225, 179,   4,  44, 120,  58,  59,  72]], dtype='uint8'),
##    '騎業火':np.array(, dtype='uint8'),
    '術業火':np.array([[ 17, 253,  38,  43,  60,  82,  26, 141]], dtype='uint8'),
    '殺業火':np.array([[169, 182,  22,  57, 254,  43,  55,  74]], dtype='uint8'),
    '狂業火':np.array([[ 73, 167, 246, 126,  58,  27, 119,  32]], dtype='uint8'),
    '全種火変換':np.array([[224, 223,  57,  15,  62,  41,   7,   5]], dtype='uint8'),
    '剣種火変換':np.array([[216, 223, 229,  53,  62,  37,  30,   8]], dtype='uint8'),
    '弓種火変換':np.array([[118,  63, 137,  53,  44,  43,  42,   5]], dtype='uint8'),
    '槍種火変換':np.array([[226,  51,  37,  45, 120,  49,  42,  64]], dtype='uint8'),
    '騎種火変換':np.array([[152, 111,  22,  22, 230,  35,  62,   8]], dtype='uint8'),
    '術種火変換':np.array([[ 92, 157,  39,  35,  60,  87,  26,  13]], dtype='uint8'),
    '殺種火変換':np.array([[249,  54,  22,  57, 184,  41,  39,  10]], dtype='uint8'),
    '狂種火変換':np.array([[ 73,  39, 246,  30,  58,  25,  55,  33]], dtype='uint8'),
    '剣灯火変換':np.array([[216, 223, 167,  49,  62,  37,  30,  10]], dtype='uint8'),
    '弓灯火変換':np.array([[118,  63, 137,  53,  44,  43,  42,   5]], dtype='uint8'),
    '槍灯火変換':np.array([[ 98, 179,  37,  45,  56,  49,  42,  72]], dtype='uint8'),
    '騎灯火変換':np.array([[152, 111,  22,  22, 166,  35,  58,   8]], dtype='uint8'),
    '術灯火変換':np.array([[ 28, 159,  39,  35,  60,  87,  26,  13]], dtype='uint8'),
    '殺灯火変換':np.array([[249,  54,  22,  57, 184,  41,  38,  10]], dtype='uint8'),
    '狂灯火変換':np.array([[ 73, 167, 246,  30,  58,  25, 247,  33]], dtype='uint8'),
    '剣大火変換':np.array([[216,  91, 229,  49,  58,  39,  62,  10]], dtype='uint8'),
    '弓大火変換':np.array([[114,  63, 137,  61, 173,  42,  42,   5]], dtype='uint8'),
    '槍大火変換':np.array([[226,  51,  37,  44, 120,  51,  43,  72]], dtype='uint8'),
    '騎大火変換':np.array([[185, 111,  54,  62, 230,  35,  63,   8]], dtype='uint8'),
    '術大火変換':np.array([[156, 141,  38,  51,  60,  86,  58,  13]], dtype='uint8'),
    '殺大火変換':np.array([[249,  38,  22,  57, 184,  41,  55,  10]], dtype='uint8'),
    '狂大火変換':np.array([[ 73,  39, 246,  62,  58,  27, 127,  33]], dtype='uint8'),
    '剣猛火変換':np.array([[216,  95, 245,  53,  62,  47,  30,  10]], dtype='uint8'),
    '弓猛火変換':np.array([[112,  63, 153,  63, 124,  43,  42,   7]], dtype='uint8'),
    '槍猛火変換':np.array([[224,  55,  21,  45, 120,  63,  58,  72]], dtype='uint8'),
    '騎猛火変換':np.array([[217, 111,  22,  54, 238,  35,  58,   8]], dtype='uint8'),
    '術猛火変換':np.array([[ 93, 253,  39,  43,  60,  86,  26,  13]], dtype='uint8'),
    '殺猛火変換':np.array([[249,  54,  86,  57, 248,  43,  39,  10]], dtype='uint8'),
    '狂猛火変換':np.array([[ 89,  39, 246,  30,  58,  27,  23,  33]], dtype='uint8'),
}

dist_local = {
}

#恒常アイテム
#ここに記述したものはドロップ数を読み込まない
#順番ルールにも使われる
# 通常 弓→槍の順番だが、種火のみ槍→弓の順番となる
# 同じレアリティの中での順番ルールは不明
std_item = ['実', 'カケラ', '卵', '鏡','炉心', '神酒', '胆石', '産毛', 'スカラベ',    
    'ランプ', '幼角', '根', '逆鱗', '心臓', '爪', '脂', '涙石' , 
    '霊子', '冠', '矢尻', '鈴', 'オーロラ',  '指輪', '結氷', '勾玉','貝殻', '勲章', 
    '八連', '蛇玉', '羽根', 'ホム', '蹄鉄', '頁', '歯車', 'ランタン', '種', 
    '火薬', '鉄杭', '髄液', '毒針', '鎖', '塵', '牙', '骨', '証', 
    '剣秘', '弓秘', '槍秘', '騎秘', '術秘', '殺秘', '狂秘',
    '剣魔', '弓魔', '槍魔', '騎魔', '術魔', '殺魔', '狂魔',
    '剣輝', '弓輝', '槍輝', '騎輝', '術輝', '殺輝', '狂輝',
    '剣モ', '弓モ', '槍モ', '騎モ', '術モ', '殺モ', '狂モ',
    '剣ピ', '弓ピ', '槍ピ', '騎ピ', '術ピ', '殺ピ', '狂ピ',
    '全種火', '全灯火', '全大火', '"全猛火','全業火',
    '剣種火', '剣灯火', '剣大火', '剣猛火', '剣業火',
    '槍種火', '槍灯火', '槍大火', '槍猛火', '槍業火',
    '弓種火', '弓灯火', '弓大火', '弓猛火', '弓業火',
    '騎種火', '騎灯火', '騎大火', '騎猛火', '騎業火',
    '術種火', '術灯火', '術大火', '術猛火', '術業火',
    '殺種火', '殺灯火', '殺大火', '殺猛火', '殺業火',
    '狂種火', '狂灯火', '狂大火', '狂猛火', '狂業火',
]
    
std_item_dic = {}
for i in std_item:
    std_item_dic[i] = 0

def imread(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8):
    """
    OpenCVのimreadが日本語ファイル名が読めない対策
    """
    try:
        n = np.fromfile(filename, dtype)
        img = cv2.imdecode(n, flags)
        return img
    except Exception as e:
        print(e)
        return None

def has_intersect(a, b):
    """
    二つの矩形の当たり判定
    隣接するのはOKとする
    """
    return max(a[0], b[0]) < min(a[2], b[2]) \
           and max(a[1], b[1]) < min(a[3], b[3])

class ScreenShot:
    """
    戦利品スクリーンショットを表すクラス
    """
    def __init__(self, img_rgb, svm, svm_chest, svm_card, debug=False, reward_only=False):
        TRAINING_IMG_WIDTH = 1755
        threshold = 80
        self.pagenum, self.pages, self.lines = pageinfo.guess_pageinfo(img_rgb)
        self.img_rgb_orig = img_rgb
        self.img_gray_orig = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        th, self.img_th_orig = cv2.threshold(self.img_gray_orig, threshold, 255, cv2.THRESH_BINARY)

        game_screen = self.extract_game_screen(debug)
        if debug:
            cv2.imwrite('game_screen.png', game_screen)

        height_g, width_g, _ = game_screen.shape
        wscale = (1.0 * width_g) / TRAINING_IMG_WIDTH
        resizeScale = 1 / wscale

        if resizeScale > 1:
            matImgResize = 1 / resizeScale
            self.img_rgb = cv2.resize(game_screen, (0,0), fx=resizeScale, fy=resizeScale, interpolation=cv2.INTER_CUBIC)
            self.img_th_resize = cv2.resize(self.img_th_orig, (0,0), fx=resizeScale, fy=resizeScale, interpolation=cv2.INTER_CUBIC)
        else:
            self.img_rgb = cv2.resize(game_screen, (0,0), fx=resizeScale, fy=resizeScale, interpolation=cv2.INTER_AREA)
            self.img_th_resize = cv2.resize(self.img_th_orig, (0,0), fx=resizeScale, fy=resizeScale, interpolation=cv2.INTER_AREA)

        if debug:
            cv2.imwrite('game_screen_resize.png', self.img_rgb)

        self.img_gray = cv2.cvtColor(self.img_rgb, cv2.COLOR_BGR2GRAY)
        th, self.img_th = cv2.threshold(self.img_gray, threshold, 255, cv2.THRESH_BINARY)
        self.svm = svm
        self.svm_chest = svm_chest
        
        self.height, self.width = self.img_rgb.shape[:2]
        self.chestnum = self.ocr_tresurechest()
 
        item_pts = []
        if self.chestnum >= 0:
            item_pts = self.img2points()

        self.items = []
        if reward_only == True:
            # qpsplit.py で利用
            item_pts = item_pts[0:1]
        for i, pt in enumerate(item_pts):
            item_img_rgb = self.img_rgb[pt[1] :  pt[3],  pt[0] :  pt[2]]
            item_img_gray = self.img_gray[pt[1] :  pt[3],  pt[0] :  pt[2]]
            self.items.append(Item(item_img_rgb, item_img_gray, svm, svm_card, debug))
            if debug:
                cv2.imwrite('item' + str(i) + '.png', item_img_rgb)
                
        if reward_only == True:
            self.reward = self.makereward()
            return

        self.itemlist = self.makelist()
        self.itemdic = dict(Counter(self.itemlist))
        self.reward = self.makereward()
        self.allitemlist = self.makelallist()
        self.allitemdic = dict(Counter(self.allitemlist))
        self.qplist = self.makeqplist()
        self.qpdic =dict(Counter(self.qplist))
        self.reisoulist = self.makereisoulist()
        self.reisoudic =dict(Counter(self.reisoulist))
        # 複数ファイル対応のためポイントはその都度消す
        if "ポイント" in dist_local.keys():
            del dist_local["ポイント"]

    def extract_game_screen(self, debug=False):
        """
        1. Make cutting image using edge and line detection
        2. Correcting to be a gamescreen from cutting image
        """
        # 1. Edge detection
        height, width = self.img_gray_orig.shape[:2]
        canny_img = cv2.Canny(self.img_gray_orig, 100, 100)

        if debug:
            cv2.imwrite("canny_img.png",canny_img)

        # 2. Line detection
        # In the case where minLineLength is too short, it catches the line of the item.
        # Some pictures fail when maxLineGap =7
        lines = cv2.HoughLinesP(canny_img, rho=1, theta=np.pi/2, threshold=80, minLineLength=int(height/5), maxLineGap=8)

        left_x = upper_y =  0
        right_x = width
        bottom_y = height
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Detect Left line
            if x1 == x2 and x1 < width/2:
                if left_x < x1: left_x = x1
            # Detect Upper line
            if y1 == y2 and y1 < height/2:
                if upper_y < y1: upper_y = y1

        # Detect Right line
        # Avoid catching the line of the scroll bar
        if debug:
                line_img = self.img_rgb_orig.copy()
            
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if debug:
                line_img = cv2.line(line_img,(x1,y1),(x2,y2),(0,0,255),1)
                cv2.imwrite("line_img.png", line_img)
            if x1 == x2 and x1 > width*3/4 and (y1 < upper_y or y2 < upper_y):
                if right_x > x1: right_x = x1

        # Detect Bottom line
        # Changed the underline of cut image to use the top of Next button.
        for line in lines:
            x1, y1, x2, y2 = line[0]
##            if y1 == y2 and y1 > height/20 and (x1 > right_x or x2 > right_x):
            if y1 == y2 and y1 > height/2 and (x1 > right_x or x2 > right_x):
                if bottom_y > y1: bottom_y = y1


        if debug:
            tmpimg = self.img_rgb_orig[upper_y:bottom_y,left_x:right_x]
            cv2.imwrite("cutting_img.png",tmpimg)

        # Correcting to be a gamescreen
        # Actual iPad (2048x1536) measurements
        scale = bottom_y - upper_y
        upper_y = upper_y - int(177*scale/847)
        bottom_y = bottom_y + int(124*scale/847)

        game_screen = self.img_rgb_orig[upper_y:bottom_y,left_x:right_x]
        return game_screen

    def makelist(self):
        """
        QPと礼装以外のアイテムを出力
        """
        itemlist = []
        for i, item in enumerate(self.items):
            if item.name[-1].isdigit():
                name = item.name + '_'
            else:
                name = item.name
            if item.card == "Point":
                std_item_dic[name + item.dropnum] = 0
            if name != 'QP' and not item.card == "Craft Essence":
                itemlist.append(name + item.dropnum)
        return itemlist

    def makeqplist(self):
        """
        Quest RewardのQP以外のQPを出力
        """
        qplist = []
        for i, item in enumerate(self.items):
##            if i != 0 and item.name == 'QP':
            if i == 0 and self.pagenum == 1:
                continue
            if  item.name == 'QP':
                qplist.append(item.name + item.dropnum)
        return qplist

    def makereisoulist(self):
        """
        礼装を出力
        """
        reisoulist = []
        for i, item in enumerate(self.items):
            if item.card == "Craft Essence":
                reisoulist.append(item.name)
        return reisoulist

    def makelallist(self):
        """
        アイテムを出力
        """
        itemlist = []
        for i, item in enumerate(self.items):
            if item.card == 'Craft Essence' or not item.name[-1].isdigit():
                name = item.name
            else:
                name = item.name + '_'
            itemlist.append(name + item.dropnum)
        return itemlist

    def makereward(self):
        """
        Quest RewardのQPを出力
        """
        if len(self.items) != 0 and self.pagenum == 1:
            return self.items[0].name + self.items[0].dropnum
        return ""
            

    def ocr_tresurechest(self):
        """
        宝箱数をOCRする関数
        """

        tb_max = 70 #宝箱数の上限値(推測)
        pt = [1443, 20, 1505, 61]
        img_num = self.img_th[pt[1]:pt[3],pt[0]:pt[2]]
        im_th = cv2.bitwise_not(img_num)
        h, w = im_th.shape[:2]

        #情報ウィンドウが数字とかぶった部分を除去する
        for y in range(h):
            im_th[y, 0] = 255
        for x in range(w): # ドロップ数7のときバグる対策 #54
            im_th[0, x] = 255
        # 物体検出
        contours = cv2.findContours(im_th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
        item_pts = []
        for cnt in contours:
            ret = cv2.boundingRect(cnt)

            pt = [ ret[0], ret[1], ret[0] + ret[2], ret[1] + ret[3] ]
            if ret[2] < int(w/2):
                flag = False
                for p in item_pts:
                    if has_intersect(p, pt) == True:
                    # どちらかを消す
                        p_area = (p[2]-p[0])*(p[3]-p[1])
                        pt_area = ret[2]*ret[3]
                        if p_area < pt_area:
                            item_pts.remove(p)                        
                        else:
                            flag = True

                if flag == False:
                    item_pts.append(pt)

        item_pts.sort()

        # Hog特徴のパラメータ
        win_size = (120, 60)
        block_size = (16, 16)
        block_stride = (4, 4)
        cell_size = (4, 4)
        bins = 9

        res = ""    
        for pt in item_pts:
            test = []
    
            tmpimg = im_th[pt[1]:pt[3], pt[0]:pt[2]]
            tmpimg = cv2.resize(tmpimg, (win_size))
            hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, bins)
            test.append(hog.compute(tmpimg)) # 特徴量の格納
            test = np.array(test)

            pred = self.svm_chest.predict(test)
            res = res + str(int(pred[1][0][0]))
         
        return int(res)

    def calc_offset(self, pts, std_pts, margin_x):
        """
        オフセットを反映
        """
        ## Y列でソート
        pts.sort(key=lambda x: x[1])
        ## Offsetを算出
        offset_x = pts[0][0] -margin_x 
        offset_y = pts[0][1] - std_pts[0][1]
        if offset_y > (std_pts[7][1] - std_pts[7][0])*2: #これ以上になったら三行目の座標と判断
            offset_y = pts[0][1] - std_pts[14][1]
        elif offset_y > 30: #これ以上になったら二行目の座標と判断
            offset_y = pts[0][1] - std_pts[7][1]

        ## Offset を反映
        item_pts = []
        for pt in std_pts:
            ptl = list(pt)
            ptl[0] = ptl[0] + offset_x
            ptl[1] = ptl[1] + offset_y
            ptl[3] = ptl[3] + offset_y
            ptl[2] = ptl[2] + offset_x
            item_pts.append(ptl)
##        print(offset_y)
        return item_pts

    def img2points(self):
        """
        戦利品左一列のY座標を求めて標準座標とのずれを補正して座標を出力する
        """
        lower = np.array([0,0,80]) 
        upper = np.array([150,171,255])

        std_pts = self.booty_pts()
        
        row_size = 7 #アイテム表示最大列
        col_size = 3 #アイテム表示最大行
        margin_x = 15
        area_size_lower = 15000 #アイテム枠の面積の最小値
        img_1strow = self.img_th[0:self.height,std_pts[0][0]-margin_x:std_pts[0][2]+margin_x]
     
        # 輪郭を抽出
        contours = cv2.findContours(img_1strow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]

        leftcell_pts = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > area_size_lower and area < self.height * self.width / (row_size * col_size):
                epsilon = 0.01*cv2.arcLength(cnt,True)
                approx = cv2.approxPolyDP(cnt,epsilon,True)
                if len(approx) == 6: #六角形のみ認識
                    ret = cv2.boundingRect(cnt)
                    if ret[1] > self.height * 0.15 \
                       and ret[1] + ret[3] < self.height * 0.76: #小数の数値はだいたいの実測                
                        pts = [ ret[0], ret[1], ret[0] + ret[2], ret[1] + ret[3] ]
                        leftcell_pts.append(pts)
        item_pts = self.calc_offset(leftcell_pts, std_pts, margin_x)

        ## 頁数と宝箱数によってすでに報告した戦利品を間引く
        if self.pagenum == 1 and self.chestnum < 20:
            item_pts = item_pts[:self.chestnum+1]
        elif self.pages - self.pagenum == 0:
            item_pts = item_pts[14-(self.lines+2)%3*7:15+self.chestnum%7]

        return item_pts

    def booty_pts(self):
        """
        戦利品が出現する21の座標 [left, top, right, bottom]
        解像度別に設定
        """
        criteria_left = 102
        criteria_top = 198
        item_width = 188
        item_height = 206
        margin_width = 32
        margin_height = 21
        pts = generate_booty_pts(criteria_left, criteria_top,
                                 item_width, item_height, margin_width, margin_height)
        return pts

def generate_booty_pts(criteria_left, criteria_top, item_width, item_height, margin_width, margin_height):
    """
        ScreenShot#booty_pts() が返すべき座標リストを生成する。
        全戦利品画像が等間隔に並んでいることを仮定している。

        criteria_left ... 左上にある戦利品の left 座標
        criteria_top ... 左上にある戦利品の top 座標
        item_width ... 戦利品画像の width
        item_height ... 戦利品画像の height
        margin_width ... 戦利品画像間の width
        margin_height ... 戦利品画像間の height
    """
    pts = []
    current = (criteria_left, criteria_top, criteria_left + item_width, criteria_top + item_height)
    for j in range(3):
        # top, bottom の y座標を計算
        current_top = criteria_top + (item_height + margin_height) * j
        current_bottom = current_top + item_height
        # x座標を左端に固定
        current = (criteria_left, current_top, criteria_left + item_width, current_bottom)
        for i in range(7):
            # y座標を固定したままx座標をスライドさせていく
            current_left = criteria_left + (item_width + margin_width) * i
            current_right = current_left + item_width
            current = (current_left, current_top, current_right, current_bottom)
            pts.append(current)
    return pts


class Item:
    def __init__(self, img_rgb, img_gray, svm, svm_card, debug=False):
        self.img_rgb = img_rgb
        self.img_gray = img_gray
        self.img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)
        th, img_th = cv2.threshold(self.img_gray, 174, 255, cv2.THRESH_BINARY)
        #174じゃないとうまくいかない IMG_8666
        #170より大きくすると0が()になる場合がある(のちにエラー訂正有)
        #176にしないとうまく分割できないときがある
        self.img_th = cv2.cv2.bitwise_not(img_th)
        
        self.height, self.width = img_rgb.shape[:2]
        self.card = self.classify_card(svm_card)
        self.name = self.classify_item(img_rgb)
        if debug == True:
            if self.name not in std_item and self.card == "Item":
                print('"' + self.name + '"', end="")
                self.name = self.classify_item(img_rgb,debug)

        self.svm = svm
        if self.name not in std_item and self.card != "Craft Essence" and self.card != "Exp. UP":
            self.ocr_digit()
        else:
            self.dropnum = ""
        if self.card == "Point":
            self.make_point_dist()
        elif self.name == "ポイント":
            self.card = "Point"

    def is_silver_item(self):
        """
        銀アイテム検出
        戦利品数OCRで銀アイテム背景だけ挙動が違うので分けるため
        """
        img_hsv_top = self.img_hsv[int(38/257*self.height):int(48/257*self.height), 7:17]

        hist_s = cv2.calcHist([img_hsv_top],[1],None,[256],[0,256]) #Sのヒストグラムを計算
        # 最小値・最大値・最小値の位置・最大値の位置を取得
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(hist_s)
        if maxLoc[1] < 17:
            return True
        return False

    def conflictcheck(self, pts, pt):
        """
        pt が ptsのどれかと衝突していたら面積に応じて入れ替える
        """
        flag = False
        for p in list(pts):
            if has_intersect(p, pt) == True:
            # どちらかを消す
                p_area = (p[2]-p[0])*(p[3]-p[1])
                pt_area = (pt[2]-pt[0])*(pt[3]-pt[1])
                if p_area < pt_area:
                    pts.remove(p)                        
                else:
                    flag = True

        if flag == False:
            pts.append(pt)
        return pts

    def extension(self, pts):
        """
        文字エリアを1pixcel微修正
        """
        new_pts = []
        for pt in pts:
            if pt[0] == 0 and pt[1] == 0:
                pt = [pt[0], pt[1], pt[2], pt[3]+1]
            elif pt[0] == 0 and pt[1] != 0:
                pt = [pt[0], pt[1] -1, pt[2], pt[3]+1]
            elif pt[0] != 0 and pt[1] == 0:
                pt = [pt[0]-1, pt[1], pt[2], pt[3]+1]
            else:
                pt = [pt[0]-1, pt[1] -1, pt[2], pt[3]+1]
            new_pts.append(pt)
        return new_pts

    def extension_straighten(self, pts):
        """
        Y軸を最大値にそろえつつ文字エリアを1pixcel微修正
        """
        base_top = 6 #強制的に高さを確保
        base_bottom = 10
        for pt in pts:
            if base_top > pt[1]:
                base_top = pt[1]
            if base_bottom < pt[3]:
                base_bottom = pt[3]

        # 5桁目がおかしくなる対策
        new_pts = []
        pts.reverse()
        for i, pt in enumerate(pts):
            if len(pts) > 6 and i == 4:
                pt = [pts[5][2], base_top, pts[3][0], base_bottom]
            else:
                pt = [pt[0], base_top, pt[2], base_bottom]
            new_pts.append(pt)
        new_pts.reverse()
        return new_pts
        
    def detect_lower_yellow_char(self):
        """
        戦利品数OCRで下段の黄文字の座標を抽出する

        HSVで黄色をマスクしてオブジェクト検出
        ノイズは少なく精度はかなり良い
        """

        margin_top = int(self.height*0.72)
        margin_bottom = int(self.height*0.11)
        margin_left = 8
        margin_right = 8
        
        img_hsv_lower = self.img_hsv[margin_top:self.height - margin_bottom,
                                     margin_left :self.width - margin_right]

        h, w = img_hsv_lower.shape[:2]
        # 手持ちスクショでうまくいっている範囲
        # 黄文字がこの数値でマスクできるかが肝
        # 未対応機種が発生したため[25,180,119] →[25,175,119]に変更
        lower_yellow = np.array([25,175,119]) 
        upper_yellow = np.array([37,255,255])

        img_hsv_lower_mask = cv2.inRange(img_hsv_lower, lower_yellow, upper_yellow)

        contours = cv2.findContours(img_hsv_lower_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]

        item_pts_lower_yellow = []
        # 物体検出マスクがうまくいっているかが成功の全て
        for cnt in contours:
            ret = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            pt = [ ret[0] + margin_left, ret[1] + margin_top,
                   ret[0] + ret[2] + margin_left, ret[1] + ret[3]  + margin_top]
            if ret[2] < int(w/2) and ret[1] < int(h*3/5) and area > 3:
                item_pts_lower_yellow = self.conflictcheck(item_pts_lower_yellow, pt)

        item_pts_lower_yellow.sort()
        if len(item_pts_lower_yellow) > 0:
            if self.width - item_pts_lower_yellow[-1][2] > int((22*self.width/188)):
                #黄文字は必ず右寄せなので最後の文字が画面端から離れている場合全部ゴミ
                item_pts_lower_yellow = []

        return self.extension(item_pts_lower_yellow)


    def get_digitimg(self, img_hsv, im_th):
        """
        白でマスク(文字のフチのみ白に)→2値化画像とOR(文字内部のみ黒に)→
        白エリア収縮することで文字(黒)拡張→反転(文字内部のみ白に)
        """
        lower_white = np.array([0,1, 0]) 
        upper_white = np.array([255,255, 255])
        img_hsv_mask = cv2.inRange(self.img_hsv, lower_white, upper_white)
        kernel = np.ones((2,1),np.uint8)
##        kernel = np.ones((1,5),np.uint8)
        img = cv2.cv2.bitwise_or(img_hsv_mask, im_th)
        erosion = cv2.erode(img,kernel,iterations = 1)
        erosion_rev = cv2.cv2.bitwise_not(erosion)

        return erosion_rev
    
    def img2digitimg(self, img_hsv, im_th):
        """
        白でマスク(文字のフチのみ白に)→2値化画像とOR(文字内部のみ黒に)→
        白エリア収縮することで文字(黒)拡張→反転(文字内部のみ白に)
        """
        lower_white = np.array([0,1, 0]) 
        upper_white = np.array([255,255, 255])
        img_hsv_mask = cv2.inRange(img_hsv, lower_white, upper_white)
        kernel = np.ones((2,2),np.uint8)
        img = cv2.cv2.bitwise_or(img_hsv_mask, im_th)
        erosion = cv2.erode(img,kernel,iterations = 1)
        erosion_rev = cv2.cv2.bitwise_not(erosion)

        return erosion_rev

    def get_font_width(self):
        """
        白文字の1文字が使用するフォント幅
        """

        if self.width == 235:
            w = 25
        elif self.width == 203 or self.width == 204:
            w = 22
        elif self.width == 188:
            w = 20
        elif self.width == 176 or self.width == 174:
            w = 18
        elif self.width == 123:
            w = 13
        else:
            w = int(220/2040*self.width)

        return w

    def get_margin_right(self):
        """
        白文字の数字右端から画面端までの幅
        """
        if self.width == 235:
            w = 20
        elif self.width == 203 or self.width == 204:
            w = 16
        elif self.width == 188:
            w = 15
        elif self.width == 176 or self.width == 174:
            w = 14
        elif self.width == 123:
            w = 10
            
        else:
            w = int(150/1880*self.width)
            
        return w

    def get_comma_width(self):
        """
        白文字のコンマ前後が使用するフォント幅
        """
        if self.width == 235:
            w = 11
        elif self.width == 203 or self.width == 204:
            w = 10
        elif self.width == 188:
            w = 9 
        elif self.width == 176 or self.width == 174:
            w = 9 
        elif self.width == 123:
            w = 6
        else:
            w = int(11/235*self.width)
            
        return w

    def detect_white_char(self, base_line, offset_x=0, cut_width=20, comma_width = 9):
        """
        上段と下段の白文字を見つける機能を一つに統合
        """

        ## QP,ポイントはボーナス6桁のときに高さが変わる
        ## それ以外は3桁のときに変わるはず(未確認)
        # この top_y は何？
#        top_y = base_line- int(240/1930*self.height)
        top_y = base_line - 26
##        cut_width = 20
        margin_right = 15
##        comma_width = 9

        if (self.name in ['QP', 'ポイント'] and len(self.dropnum) > 6 + 2) or \
           (self.name not in ['QP', 'ポイント'] and len(self.dropnum) > 3 + 2):
            # 1. 下2桁から文字の高さを推測
            top_y = base_line- 23
            digitimg = self.get_digitimg(self.img_hsv, self.img_th)
            #下2桁切り出して上下幅を決める

            img_right = digitimg[top_y:base_line+1,
                                 int(870/1230*self.width) - offset_x:self.width - margin_right - offset_x]
            hr, wr = img_right.shape[:2]
            for x in range(wr):
                img_right[0, x] = 0
                img_right[hr-1, x] = 0
            for y in range(hr):
                img_right[y, 0] = 0
                img_right[y, wr-1] = 0

            contours = cv2.findContours(img_right, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            tmp_pts = []
            for cnt in contours:
                ret = cv2.boundingRect(cnt)
                area = cv2.contourArea(cnt)
                pt = [ ret[0], ret[1], ret[0] + ret[2], ret[1] + ret[3] ]
                if area > int(23/135*self.height) and ret[2] > int((60/1240)*self.width): #この数値は結構デリケート
                    tmp_pts = self.conflictcheck(tmp_pts, pt)
            tmp_pts.sort()
            if len(tmp_pts) > 0:
                top_y = top_y + (tmp_pts[-1][1] - int(10/1350*self.height))
            if len(tmp_pts) == 1:
                cut_width = int(18/24*(tmp_pts[0][3] - tmp_pts[0][1] + int(40/1240*self.width)))
            elif len(tmp_pts) > 1:
                cut_width = int(tmp_pts[-1][2]-tmp_pts[-2][2])

        ## まず、+, xの位置が何桁目か調査する
        for i in range(8): #8桁以上は無い
            if i == 0:
                continue
            pt = [self.width - margin_right - cut_width * (i + 1) - comma_width * int((i - 1)/3) - offset_x,
                  top_y,
                  self.width - margin_right  - cut_width * i  - comma_width * int((i  - 1)/3) - offset_x,
                  base_line]
            result = self.read_char(pt)
            if i == 1 and ord(result) == 0:
                # アイテム数 x1 とならず表記無し場合のエラー処理
                return ""
            if result in ['x', '+']:
                break
        ## 決まった位置まで出力する
        line = ""
        for j in range(i): 
            pt = [self.width - margin_right - cut_width * (j + 1) - comma_width * int((j + 1)/4) - offset_x,
                  top_y,
                  self.width - margin_right  - cut_width * j  - comma_width * int((j + 1)/4) - offset_x,
                  base_line]
            c = self.read_char(pt)
            line = line + c
        j = j + 1
        pt = [self.width - margin_right - cut_width * (j + 1) - comma_width * int((j - 1)/3) - offset_x,
            top_y,
            self.width - margin_right  - cut_width * j  - comma_width * int((j  - 1)/3) - offset_x,
            base_line]
        c = self.read_char(pt)
        line = line + c

        line = line[::-1]
        line = re.sub('000000$', "百万", line)
        line = re.sub('0000$', "万", line)
        if len(line) > 5:
            line = re.sub('000$', "千", line)

        return line

    def read_item(self, pts, upper=False, yellow=False,):
        """
        戦利品の数値をOCRする(エラー訂正有)
        """

        win_size = (120, 60)
        block_size = (16, 16)
        block_stride = (4, 4)
        cell_size = (4, 4)
        bins = 9
        lines = ""

        for pt in pts:
            char = []
            tmpimg = self.img_gray[pt[1]:pt[3], pt[0]:pt[2]]
            
            tmpimg = cv2.resize(tmpimg, (win_size))
            hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, bins)
            char.append(hog.compute(tmpimg))
            char = np.array(char)
            pred = self.svm.predict(char)
            result = int(pred[1][0][0])
            if result != 0:
                lines = lines + chr(result)
        #以下エラー訂正
        if yellow==True:
            if not lines.endswith(")") or "(+" not in lines:
                lines = ""
        lines = lines.replace("()", "0")
        if len(lines) > 1:
            #エラー訂正 文字列左側
            # 主にイベントのポイントドロップで左側にゴミができるが、
            # 特定の記号がでてきたらそれより前はデータが無いはずなので削除する
            point_lbra = lines.rfind("(")
            point_plus = lines.rfind("+")
            point_x = lines.rfind("x")
            if yellow==True and point_lbra != -1:
                lines = lines[point_lbra:]
            elif point_plus != -1:
                lines = lines[point_plus:]
            elif point_x != -1:
                lines = lines[point_x:]

        if upper == True:
            #エラー訂正 文字列右側
            # イベントでポイントがドロップするとき、ポイントは銀枠で
            # 文字の右側が抜け落ちるが抜け落ちた部分は 0 なので訂正可能
            if len(pts) > 0:
                for i in range(int((self.width-pts[-1][2])/(21/190*self.width))):
                    lines = lines + '0'
            if lines.isdigit():
                lines = '+' + lines
        else:
            if lines.isdigit():
                if int(lines) == 0:
                    lines = "xErr"
                elif self.name == "QP":
                    lines = '+' + lines
                else:
                    if int(lines) >= 100:
                        lines = '+' + lines
                    else:
                        lines = 'x' + lines

        if len(lines) == 1:
            lines = "xErr"

        return lines

    def read_char(self, pt):
        """
        戦利品の数値1文字をOCRする
        白文字検出で使用
        """
        win_size = (120, 60)
        block_size = (16, 16)
        block_stride = (4, 4)
        cell_size = (4, 4)
        bins = 9
        lines = ""
        char = []
        tmpimg = self.img_gray[pt[1]:pt[3], pt[0]:pt[2]]

        tmpimg = cv2.resize(tmpimg, (win_size))
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, bins)
        char.append(hog.compute(tmpimg))
        char = np.array(char)
        pred = self.svm.predict(char)
        result = int(pred[1][0][0])
        return chr(result)

    def ocr_digit(self):
        """
        戦利品OCR
        """
        cut_width = 20
        comma_width = 9
        flag_silver = False
        if self.is_silver_item() == True:
            flag_silver = True

        item_pts_lower_yellow = self.detect_lower_yellow_char()        
        self.dropnum = self.read_item(item_pts_lower_yellow, yellow=True)
        if self.name in ["QP", "ポイント"] and len(self.dropnum) >= 5: #ボーナスは"(+*0)"なので
            # 末尾の括弧上部からの距離を設定
            base_line = item_pts_lower_yellow[-1][1] -int(4/206*self.height)
            if len(self.dropnum) >= 9: # issue: #57
                cut_width = 18
                comma_width = 8
        else:
            base_line = int(180/206*self.height)

        if self.name in ["QP", "ポイント"]:
            x = 0            
        elif len(item_pts_lower_yellow) > 0:
            x = self.width - (item_pts_lower_yellow[0][0] -0) - int(130/1880*self.width)
        else:
            x = 0
        self.dropnum =  self.detect_white_char(base_line, offset_x = x, cut_width = cut_width, comma_width = comma_width) + self.dropnum
        
        self.dropnum =re.sub("\([^\(\)]*\)$", "", self.dropnum) #括弧除去
        if self.dropnum != "":
            self.dropnum = "(" + self.dropnum + ")"

    def classify_standard_item(self, img, debug=False):
        """
        imgとの距離を比較して近いアイテムを求める
        """
        # 種火かどうかの判別
        hash_tanebi = self.compute_tanebi_hash(img)
        tanebifiles = {}
        for i in dist_tanebi.keys():
            dt = hasher.compare(hash_tanebi, dist_tanebi[i])
            if dt <= 15: #IMG_1833で11 IMG_1837で15
                tanebifiles[i] = dt
        tanebifiles = sorted(tanebifiles.items(), key=lambda x:x[1])

        if len(tanebifiles) > 0:
            tanebi = next(iter(tanebifiles))
            hash_tanebi_class = self.compute_tanebi_class_hash(img)
            tanebiclassfiles = {}
            for i in dist_tanebi_class.keys():
                if (tanebi[0].replace('変換', ''))[-2:] in i:
                    dtc = hasher.compare(hash_tanebi_class, dist_tanebi_class[i])
                    if dtc <= 19: #18離れることがあったので(Screenshot_20200318-140020.png)
                        tanebiclassfiles[i] = dtc
            tanebiclassfiles = sorted(tanebiclassfiles.items(), key=lambda x:x[1])
            if len(tanebiclassfiles) > 0:
                tanebiclass = next(iter(tanebiclassfiles))
                return tanebiclass[0].replace('変換', '')
        
        hash_item = compute_hash(img) #画像の距離
        itemfiles = {}
        if debug == True:
            print(":np.array([" + str(list(hash_item[0])) + "], dtype='uint8'),")
        # 既存のアイテムとの距離を比較
        for i in dist_item.keys():
            d = hasher.compare(hash_item, dist_item[i])
            if d <= 10:
            #ポイントと種の距離が8という例有り(IMG_0274)→16に
            #バーガーと脂の距離が10という例有り(IMG_2354)→14に
                itemfiles[i] = d
        if len(itemfiles) > 0:
            itemfiles = sorted(itemfiles.items(), key=lambda x:x[1])
            item = next(iter(itemfiles))
            if item[0].endswith("秘"):
                hash_hi = self.compute_maseki_hash(img)
                hisekifiles = {}
                for i in dist_hiseki.keys():
                    d2 = hasher.compare(hash_hi, dist_hiseki[i])
                    if d2 <= 20:
                        hisekifiles[i] = d2
                hisekifiles = sorted(hisekifiles.items(), key=lambda x:x[1])
                item = next(iter(hisekifiles))
            elif item[0].endswith("魔"):
                hash_ma = self.compute_maseki_hash(img)
                masekifiles = {}
                for i in dist_maseki.keys():
                    d2 = hasher.compare(hash_ma, dist_maseki[i])
                    if d2 <= 20:
                        masekifiles[i] = d2
                masekifiles = sorted(masekifiles.items(), key=lambda x:x[1])
                item = next(iter(masekifiles))
            elif item[0].endswith("輝"):
                hash_ki = self.compute_maseki_hash(img)
                kisekifiles = {}
                for i in dist_kiseki.keys():
                    d2 = hasher.compare(hash_ki, dist_kiseki[i])
                    if d2 <= 20:
                        kisekifiles[i] = d2
                kisekifiles = sorted(kisekifiles.items(), key=lambda x:x[1])
                item = next(iter(kisekifiles))
            elif item[0].endswith("モ") or item[0].endswith("ピ"):
                #ヒストグラム
                img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                h, w = img_hsv.shape[:2]
                img_hsv = img_hsv[int(h/2-10):int(h/2+10),int(w/2-10):int(w/2+10)]
                hist_s = cv2.calcHist([img_hsv],[1],None,[256],[0,256]) #Bのヒストグラムを計算
                minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(hist_s)
                if maxLoc[1] > 128:
                    return item[0][0] + "モ"
                else:
                    return item[0][0] + "ピ"
                
            return item[0]

        return ""

    def classify_local_item(self, img):
        """
        既所持のアイテム画像の距離を計算して保持
        """
        hash_item = compute_hash(img) #画像の距離

        itemfiles = {}
        # 既存のアイテムとの距離を比較
        for i in dist_local.keys():
            d = hasher.compare(hash_item, dist_local[i])
            #同じアイテムでも14離れることあり(IMG_8785)
            if d <= 15:
                itemfiles[i] = d
        if len(itemfiles) > 0:
            itemfiles = sorted(itemfiles.items(), key=lambda x:x[1])
            item = next(iter(itemfiles))
            if type(item[0]) is str: #ポイント登録用
                return item[0]
            return item[0].stem

        return ""

    def classify_tanebi(self, img):
        hash_item = self.compute_tanebi_rarity_hash(img) #画像の距離
        itemfiles = {}
        for i in dist_tanebi_rarity.keys():
            d = hasher.compare(hash_item, dist_tanebi_rarity[i])
            itemfiles[i] = d
        itemfiles = sorted(itemfiles.items(), key=lambda x:x[1])
        item = next(iter(itemfiles))
        hash_tanebi_class = self.compute_tanebi_class_hash(img)
        tanebiclassfiles = {}
        for i in dist_tanebi_class.keys():
            dtc = hasher.compare(hash_tanebi_class, dist_tanebi_class[i])
            tanebiclassfiles[i] = dtc
        tanebiclassfiles = sorted(tanebiclassfiles.items(), key=lambda x:x[1])
        tanebiclass = next(iter(tanebiclassfiles))

        result = tanebiclass[0][0] + item[0].replace('変換', '')
        return result

    def make_point_dist(self):
        """
        3行目に現れ、Point表示が削れているアイテムのために
        Pointを登録しておく
        """
        if "ポイント" not in dist_local.keys():
            dist_local["ポイント"] = compute_hash(self.img_rgb) #画像の距離

        
    def make_new_file(self, img):
        """
        ファイル名候補を探す
        """
        for i in range(99999):
            itemfile = Item_dir / ('item{:0=6}'.format(i + 1) + '.png')
            if itemfile.is_file():
                continue
            else:
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
                cv2.imwrite(itemfile.as_posix(), img_gray)
                dist_local[itemfile] = compute_hash(img)
                break
        return itemfile.stem

    def make_new_file_gray(self, img):
        """
        ファイル名候補を探す
        """
        for i in range(99999):
            itemfile = Item_dir / ('item{:0=6}'.format(i + 1) + '.png')
            if itemfile.is_file():
                continue
            else:
##                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
                cv2.imwrite(itemfile.as_posix(), img)
                dist_local[itemfile] = compute_hash(img)
                break
        return itemfile.stem


    def classify_card(self, svm_card):
        """
        カード判別器
       """
        """
        カード判別器
        この場合は画像全域のハッシュをとる
        """
        # Hog特徴のパラメータ
        win_size = (120, 60)
        block_size = (16, 16)
        block_stride = (4, 4)
        cell_size = (4, 4)
        bins = 9
        test = []
        carddic = { 0:'Quest Reward', 1:'Item', 2:'Point', 3:'Craft Essence', 4:'Exp. UP', 99:"" }

        tmpimg = self.img_rgb[int(189/206*self.height):int(201/206*self.height),
                      int(78/188*self.width):int(115/188*self.width)]
        
        tmpimg = cv2.resize(tmpimg, (win_size))
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, bins)
        test.append(hog.compute(tmpimg)) # 特徴量の格納
        test = np.array(test)
        pred = svm_card.predict(test)

        return carddic[pred[1][0][0]]
        
    def classify_item(self, img, debug=False):
        """
        アイテム判別器
        """
        if self.card == "Point":
            return "ポイント"
        elif self.card == "Quest Reward":
            return "QP"
        elif self.card == "Exp. UP":
            return self.classify_tanebi(img)
        item = self.classify_standard_item(img, debug)
        if item == "":
            item = self.classify_local_item(img)
        if item == "":
            item = self.make_new_file(img)
        return item

    def compute_tanebi_hash(self, img_rgb):
        """
        種火レアリティ判別器
        この場合は画像全域のハッシュをとる
        """
        return hasher.compute(img_rgb)

    def compute_tanebi_rarity_hash(self, img_rgb):
        """
        種火レアリティ判別器
        この場合は画像全域のハッシュをとる
        """
        img = img_rgb[int(53/189*self.height):int(136/189*self.height),
                      int(37/206*self.width):int(149/206*self.width)]
        return hasher.compute(img)

    def compute_tanebi_class_hash(self, img_rgb):
        """
        種火クラス判別器
        左上のクラスマークぎりぎりのハッシュを取る
        記述した比率はiPhone6S画像の実測値
        """
        img = img_rgb[int(5/135*self.height):int(30/135*self.height),
                      int(5/135*self.width):int(30/135*self.width)]
        return hasher.compute(img)

    def compute_maseki_hash(self, img_rgb):
        """
        魔石クラス判別器
        中央のクラスマークぎりぎりのハッシュを取る
        記述した比率はiPhone6S画像の実測値
        """
        img = img_rgb[int(41/135*self.height):int(84/135*self.height),
                      int(44/124*self.width):int(79/124*self.width)]
        return hasher.compute(img)

def compute_hash(img_rgb):
    """
    判別器
    この判別器は下部のドロップ数を除いた部分を比較するもの
    記述した比率はiPhone6S画像の実測値
    """
    height, width = img_rgb.shape[:2]
    img = img_rgb[int(17/135*height):int(77/135*height),
                    int(19/135*width):int(103/135*width)]
    return hasher.compute(img)
        
def calc_dist_local():
    """
    既所持のアイテム画像の距離(一次元配列)の辞書を作成して保持
    """
    files = Item_dir.glob('**/*.png')
    for fname in files:
        img = imread(fname)
        dist_local[fname] = compute_hash(img)


def get_output(filenames, debug=False):
    """
    出力内容を作成
    """
    calc_dist_local()
    if train_item.exists() == False:
        print("[エラー]item.xml が存在しません")
        print("python makeitem.py を実行してください")
        sys.exit(1)
    if train_chest.exists() == False:
        print("[エラー]chest.xml が存在しません")
        print("python makechest.py を実行してください")
        sys.exit(1)
    if train_card.exists() == False:
        print("[エラー]card.xml が存在しません")
        print("python makecard.py を実行してください")
        sys.exit(1)
    svm = cv2.ml.SVM_load(str(train_item))
    svm_chest = cv2.ml.SVM_load(str(train_chest))
    svm_card = cv2.ml.SVM_load(str(train_card))

    csvfieldnames = { 'filename' : "合計", 'ドロ数': "" } #CSVフィールド名用 key しか使わない
    wholelist = []
    rewardlist = []
    reisoulist = []
    qplist = []
    outputcsv = [] #出力
    prev_pages = 0
    prev_pagenum = 0

    for filename in filenames:
        if debug:
            print(filename)
        f = Path(filename)

        if f.exists() == False:
            output = { 'filename': filename + ': Not Found' }
        else:            
            img_rgb = imread(filename)

            try:
                sc = ScreenShot(img_rgb, svm, svm_chest, svm_card, debug)

                #2頁目以降のスクショが無い場合に migging と出力                
                if (prev_pages - prev_pagenum > 0 and sc.pagenum - prev_pagenum != 1) \
                   or (prev_pages - prev_pagenum == 0 and sc.pagenum != 1):
                    outputcsv.append({'filename': 'missing'})
                    
                prev_pages = sc.pages
                prev_pagenum = sc.pagenum

                #戦利品順番ルールに則った対応による出力処理
                wholelist = wholelist + sc.itemlist
                if sc.reward != "":
                    rewardlist = rewardlist + [sc.reward]
                reisoulist = reisoulist + sc.reisoulist
                qplist = qplist + sc.qplist
                output = { 'filename': filename, 'ドロ数':len(sc.itemlist) + len(sc.qplist) + len(sc.reisoulist) }
                output.update(sc.allitemdic)
                if sc.pagenum == 1:
                    if sc.lines >= 7:
                        output["ドロ数"] = str(output["ドロ数"]) + "++"
                    elif sc.lines >= 4:
                        output["ドロ数"] = str(output["ドロ数"]) + "+"
                elif sc.pagenum == 2 and sc.lines >= 7:             
                    output["ドロ数"] = str(output["ドロ数"]) + "+"
                output.update(sc.allitemdic)
            except:
                output = ({'filename': str(filename) + ': not valid'})
        outputcsv.append(output)

    csvfieldnames.update(dict(Counter(rewardlist)))
    reisou_dic = dict(Counter(reisoulist))
    csvfieldnames.update(sorted(reisou_dic.items(), reverse=True))
 
    std_item_dic.update(dict(Counter(wholelist)))
    qp_dic = dict(Counter(qplist))
    
    for key in list(std_item_dic.keys()):
        if std_item_dic[key] == 0:
            del std_item_dic[key]
    csvfieldnames.update(std_item_dic)
    csvfieldnames.update(sorted(qp_dic.items()))

    return csvfieldnames, outputcsv

if __name__ == '__main__':
    ## オプションの解析
    parser = argparse.ArgumentParser(description='FGOスクショからアイテムをCSV出力する')
    # 3. parser.add_argumentで受け取る引数を追加していく
    parser.add_argument('filenames', help='入力ファイル', nargs='*')    # 必須の引数を追加
    parser.add_argument('-f', '--folder', help='フォルダで指定')
    parser.add_argument('-d', '--debug', help='デバッグ情報の出力', action='store_true')
    parser.add_argument('--version', action='version', version=progname + " " + version)

    args = parser.parse_args()    # 引数を解析

    if not Item_dir.is_dir():
        Item_dir.mkdir()

    if args.folder:
        inputs = [x for x in Path(args.folder).iterdir()]
    else:
        inputs = args.filenames
    csvfieldnames, outputcsv = get_output(inputs, args.debug)
    
    fnames = csvfieldnames.keys()
    writer = csv.DictWriter(sys.stdout, fieldnames=fnames, lineterminator='\n')
    writer.writeheader()
    if len(outputcsv) > 1: #ファイル一つのときは合計値は出さない
        writer.writerow(csvfieldnames)
    for o in outputcsv:
        writer.writerow(o)
    if 'ドロ数' in o.keys(): # issue: #55
        if len(outputcsv) > 1 and str(o['ドロ数']).endswith('+'):
            writer.writerow({'filename': 'missing'})
