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
version = "0.1.2"

Item_dir = Path(__file__).resolve().parent / Path("item/")
train_item = Path(__file__).resolve().parent / Path("item.xml") #アイテム下部
train_chest = Path(__file__).resolve().parent / Path("chest.xml") #ドロップ数
train_card = Path(__file__).resolve().parent / Path("card.xml") #ドロップ数

hasher = cv2.img_hash.PHash_create()

#恒常アイテムのハッシュ値
dist_item ={
    'QP':np.array([[82, 9, 116, 163, 228, 2, 241, 9]], dtype='uint8'),
    '爪':np.array([[254, 3, 81, 60, 13, 198, 51, 18]], dtype='uint8'),
    '心臓':np.array([[94, 131, 49, 137, 76, 76, 0, 90]], dtype='uint8'),
    '逆鱗':np.array([[142, 105, 57, 145, 231, 7, 78, 75]], dtype='uint8'),
    '根':np.array([[86, 41, 169, 73, 180, 22, 41, 9]], dtype='uint8'),
    '幼角':np.array([[146, 41, 86, 41, 214, 73, 165, 210]], dtype='uint8'),
    '涙石':np.array([[150, 33, 88, 166, 104, 18, 153, 100]], dtype='uint8'),
    '脂':np.array([[30, 5, 161, 233, 27, 88, 20, 132]], dtype='uint8'),
    'ランプ':np.array([[182, 41, 197, 112, 21, 204, 131, 120]], dtype='uint8'),
    'スカラベ':np.array([[190, 13, 67, 113, 24, 144, 132, 16]], dtype='uint8'),
    '産毛':np.array([[30, 35, 153, 41, 164, 22, 146, 74]], dtype='uint8'),
    '胆石':np.array([[22, 129, 33, 178, 156, 99, 38, 148]], dtype='uint8'),
    '神酒':np.array([[30, 132, 37, 169, 88, 73, 100, 110]], dtype='uint8'),
    '炉心':np.array([[122, 5, 36, 171, 9, 88, 17, 136]], dtype='uint8'),
    '鏡':np.array([[254, 3, 196, 86, 162, 164, 41, 41]], dtype='uint8'),
    '卵':np.array([[22, 161, 237, 88, 25, 9, 73, 8]], dtype='uint8'),
    'カケラ':np.array([[30, 129, 102, 88, 5, 154, 5, 44]], dtype='uint8'),
    '種':np.array([[30, 197, 169, 41, 90, 148, 134, 32]], dtype='uint8'),
    'ランタン':np.array([[166, 201, 25, 154, 166, 100, 121, 38]], dtype='uint8'),
    '八連':np.array([[126, 5, 165, 201, 25, 150, 98, 36]], dtype='uint8'),
    '宝玉':np.array([[86, 165, 195, 114, 185, 177, 137, 50]], dtype='uint8'),
    '羽根':np.array([[134, 187, 77, 68, 105, 20, 68, 162]], dtype='uint8'),
    '歯車':np.array([[94, 18, 225, 61, 25, 141, 12, 228]], dtype='uint8'),
    '頁':np.array([[223, 137, 246, 28, 178, 236, 91, 184]], dtype='uint8'),
    'ホム':np.array([[86, 225, 74, 132, 177, 25, 173, 37]], dtype='uint8'),
    '蹄鉄':np.array([[120, 133, 115, 185, 24, 196, 100, 34]], dtype='uint8'),
    '勲章':np.array([[150, 82, 109, 173, 181, 20, 108, 43]], dtype='uint8'),
    '貝殻':np.array([[26, 165, 213, 72, 140, 214, 176, 73]], dtype='uint8'),
    '勾玉':np.array([[254, 5, 98, 233, 92, 182, 13, 204]], dtype='uint8'),
    '結氷':np.array([[126, 129, 108, 198, 147, 106, 201, 54]], dtype='uint8'),
    '指輪':np.array([[122, 7, 197, 177, 123, 11, 81, 24]], dtype='uint8'),
    'オーロラ':np.array([[94, 163, 85, 20, 169, 137, 36, 105]], dtype='uint8'),
    '鈴':np.array([[122, 5, 193, 131, 60, 42, 82, 22]], dtype='uint8'),
    '矢尻':np.array([[190, 1, 104, 168, 77, 66, 180, 130]], dtype='uint8'),
    '冠':np.array([[233, 13, 122, 193, 62, 197, 250, 233]], dtype='uint8'),
    '証':np.array([[94, 5, 161, 88, 134, 86, 33, 25]], dtype='uint8'),
    '骨':np.array([[82, 75, 37, 149, 85, 33, 168, 165]], dtype='uint8'),
    '牙':np.array([[58, 131, 21, 217, 229, 101, 44, 176]], dtype='uint8'),
    '塵':np.array([[222, 1, 120, 30, 134, 164, 32, 17]], dtype='uint8'),
    '鎖':np.array([[14, 83, 49, 25, 204, 169, 38, 22]], dtype='uint8'),
    '毒針':np.array([[90, 181, 41, 75, 211, 178, 52, 108]], dtype='uint8'),
    '髄液':np.array([[38, 25, 114, 197, 154, 7, 2, 100]], dtype='uint8'),
    '鉄杭':np.array([[206, 141, 51, 99, 227, 99, 48, 48]], dtype='uint8'),
    '火薬':np.array([[110, 147, 57, 157, 69, 53, 194, 42]], dtype='uint8'),
    '剣秘':np.array([[106, 150, 230, 97, 31, 216, 205, 186]], dtype='uint8'),
    '弓秘':np.array([[104, 22, 102, 227, 31, 216, 205, 106]], dtype='uint8'),
    '槍秘':np.array([[232, 6, 50, 97, 159, 176, 207, 154]], dtype='uint8'),
    '騎秘':np.array([[58, 142, 118, 97, 31, 217, 207, 58]], dtype='uint8'),
    '術秘':np.array([[122, 6, 102, 51, 31, 232, 153, 110]], dtype='uint8'),
    '殺秘':np.array([[42, 150, 102, 227, 27, 232, 77, 158]], dtype='uint8'),
    '狂秘':np.array([[234, 30, 102, 226, 27, 248, 5, 158]], dtype='uint8'),
    '剣魔':np.array([[94, 161, 41, 137, 9, 72, 76, 66]], dtype='uint8'),
    '弓魔':np.array([[22, 225, 41, 41, 136, 72, 92, 66]], dtype='uint8'),
    '槍魔':np.array([[86, 161, 169, 57, 41, 72, 24, 66]], dtype='uint8'),
    '騎魔':np.array([[94, 161, 41, 25, 137, 76, 24, 70]], dtype='uint8'),
    '術魔':np.array([[94, 225, 49, 9, 9, 8, 68, 66]], dtype='uint8'),
    '殺魔':np.array([[94, 165, 185, 41, 12, 72, 76, 66]], dtype='uint8'),
    '狂魔':np.array([[94, 165, 185, 41, 12, 72, 88, 102]], dtype='uint8'),
    '剣輝':np.array([[30, 225, 41, 90, 82, 134, 164, 33]], dtype='uint8'),
    '弓輝':np.array([[30, 225, 169, 89, 214, 38, 134, 32]], dtype='uint8'),
    '槍輝':np.array([[30, 225, 169, 27, 70, 198, 166, 33]], dtype='uint8'),
    '騎輝':np.array([[30, 225, 169, 89, 86, 102, 128, 36]], dtype='uint8'),
    '術輝':np.array([[30, 225, 169, 73, 86, 150, 166, 41]], dtype='uint8'),
    '殺輝':np.array([[30, 229, 169, 89, 70, 22, 164, 32]], dtype='uint8'),
    '狂輝':np.array([[30, 229, 169, 121, 70, 150, 132, 36]], dtype='uint8'),
    '剣モ':np.array([[150, 161, 73, 73, 100, 155, 182, 38]], dtype='uint8'),
    '弓モ':np.array([[86, 153, 35, 66, 132, 25, 61, 58]], dtype='uint8'),
    '槍モ':np.array([[214, 41, 26, 73, 164, 13, 102, 146]], dtype='uint8'),
    '騎モ':np.array([[54, 233, 25, 158, 101, 58, 137, 68]], dtype='uint8'),
    '術モ':np.array([[70, 161, 24, 183, 100, 83, 156, 98]], dtype='uint8'),
    '殺モ':np.array([[102, 185, 204, 210, 37, 38, 17, 78]], dtype='uint8'),
    '狂モ':np.array([[14, 73, 179, 83, 73, 134, 100, 43]], dtype='uint8'),
    '剣ピ':np.array([[150, 185, 73, 73, 100, 154, 166, 36]], dtype='uint8'),
    '弓ピ':np.array([[86, 153, 35, 66, 132, 89, 61, 58]], dtype='uint8'),
    '槍ピ':np.array([[214, 57, 58, 200, 164, 44, 102, 146]], dtype='uint8'),
    '騎ピ':np.array([[54, 233, 25, 138, 101, 58, 137, 100]], dtype='uint8'),
    '術ピ':np.array([[70, 233, 24, 178, 100, 83, 172, 98]], dtype='uint8'),
    '殺ピ':np.array([[102, 185, 204, 210, 53, 38, 153, 78]], dtype='uint8'),
    '狂ピ':np.array([[6, 105, 163, 82, 89, 150, 116, 43]], dtype='uint8'),
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

#種火のレアリティを見分けるハッシュ値
dist_tanebi = {
    '全種火':np.array([[241, 88, 142, 178, 78, 205, 238, 43]], dtype='uint8'),
    '剣種火':np.array([[241, 40, 46, 186, 111, 249, 239, 172]], dtype='uint8'),
    '弓種火':np.array([[241, 232, 174, 186, 239, 249, 111, 172]], dtype='uint8'),
    '槍種火':np.array([[241, 232, 174, 186, 79, 253, 111, 172]], dtype='uint8'),
    '騎種火':np.array([[241, 42, 46, 186, 239, 249, 239, 172]], dtype='uint8'),
    '術種火':np.array([[241, 168, 174, 187, 15, 249, 238, 168]], dtype='uint8'),
    '殺種火':np.array([[241, 168, 38, 186, 239, 249, 238, 172]], dtype='uint8'),
    '狂種火':np.array([[241, 46, 46, 186, 239, 249, 238, 168]], dtype='uint8'),
    '剣灯火':np.array([[123, 220, 46, 99, 239, 153, 126, 237]], dtype='uint8'),
    '弓灯火':np.array([[251, 220, 46, 227, 207, 153, 254, 205]], dtype='uint8'),
    '槍灯火':np.array([[243, 220, 46, 227, 207, 155, 126, 205]], dtype='uint8'),
    '騎灯火':np.array([[123, 94, 46, 99, 239, 153, 254, 237]], dtype='uint8'),
    '術灯火':np.array([[251, 220, 46, 99, 207, 153, 126, 237]], dtype='uint8'),
    '殺灯火':np.array([[251, 204, 46, 227, 239, 185, 254, 237]], dtype='uint8'),
    '狂灯火':np.array([[115, 94, 46, 227, 239, 153, 244, 236]], dtype='uint8'),
    '剣大火':np.array([[115, 120, 10, 147, 182, 250, 43, 230]], dtype='uint8'),
    '弓大火':np.array([[243, 248, 74, 179, 164, 234, 43, 230]], dtype='uint8'),
    '槍大火':np.array([[243, 120, 10, 147, 182, 110, 47, 230]], dtype='uint8'),
    '騎大火':np.array([[115, 120, 10, 147, 182, 234, 43, 230]], dtype='uint8'),
    '術大火':np.array([[115, 120, 10, 147, 166, 110, 43, 230]], dtype='uint8'),
    '殺大火':np.array([[115, 120, 10, 147, 180, 248, 41, 230]], dtype='uint8'),
    '狂大火':np.array([[115, 120, 10, 147, 180, 232, 41, 230]], dtype='uint8'),
    '剣猛火':np.array([[11, 40, 244, 186, 190, 243, 207, 161]], dtype='uint8'),
    '弓猛火':np.array([[75, 168, 252, 186, 158, 243, 207, 162]], dtype='uint8'),
    '槍猛火':np.array([[11, 40, 244, 186, 190, 243, 207, 163]], dtype='uint8'),
    '騎猛火':np.array([[11, 8, 244, 186, 190, 243, 207, 161]], dtype='uint8'),
    '術猛火':np.array([[11, 40, 252, 187, 142, 243, 207, 163]], dtype='uint8'),
    '殺猛火':np.array([[11, 40, 244, 186, 190, 243, 207, 161]], dtype='uint8'),
    '狂猛火':np.array([[11, 42, 252, 186, 190, 243, 207, 160]], dtype='uint8'),
    '剣業火':np.array([[9, 47, 174, 248, 47, 94, 123, 175]], dtype='uint8'),
    '弓業火':np.array([[73, 47, 174, 248, 47, 94, 123, 175]], dtype='uint8'),
    '槍業火':np.array([[9, 47, 174, 120, 47, 95, 123, 171]], dtype='uint8'),
##    '騎業火':np.array(None,dtype='uint8'),
    '術業火':np.array([[9, 47, 174, 248, 47, 94, 123, 175]], dtype='uint8'),
    '殺業火':np.array([[9, 47, 174, 248, 47, 94, 123, 171]], dtype='uint8'),
    '狂業火':np.array([[9, 47, 174, 120, 47, 94, 243, 170]], dtype='uint8'),
    '全種火変換':np.array([[75, 248, 248, 7, 244, 172, 6, 150]], dtype='uint8'),
    '剣種火変換':np.array([[75, 249, 249, 15, 244, 172, 6, 150]], dtype='uint8'),
    '弓種火変換':np.array([[75, 248, 248, 15, 244, 172, 6, 150]], dtype='uint8'),
    '槍種火変換':np.array([[75, 248, 248, 15, 244, 172, 6, 150]], dtype='uint8'),
    '騎種火変換':np.array([[75, 248, 248, 15, 244, 172, 6, 150]], dtype='uint8'),
    '術種火変換':np.array([[75, 248, 248, 15, 244, 172, 6, 150]], dtype='uint8'),
    '殺種火変換':np.array([[75, 249, 249, 15, 244, 172, 6, 182]], dtype='uint8'),
    '狂種火変換':np.array([[75, 248, 248, 15, 244, 172, 6, 150]], dtype='uint8'),
    '剣灯火変換':np.array([[75, 248, 249, 7, 244, 172, 6, 166]], dtype='uint8'),
    '弓灯火変換':np.array([[75, 248, 248, 7, 244, 172, 6, 134]], dtype='uint8'),
    '槍灯火変換':np.array([[75, 248, 248, 7, 244, 172, 6, 134]], dtype='uint8'),
    '騎灯火変換':np.array([[75, 248, 248, 7, 244, 172, 6, 134]], dtype='uint8'),
    '術灯火変換':np.array([[75, 248, 248, 7, 244, 172, 6, 134]], dtype='uint8'),
    '殺灯火変換':np.array([[75, 248, 249, 7, 244, 172, 6, 182]], dtype='uint8'),
    '狂灯火変換':np.array([[75, 248, 248, 7, 244, 172, 6, 134]], dtype='uint8'),
    '剣大火変換':np.array([[7, 249, 248, 191, 166, 143, 46, 244]], dtype='uint8'),
    '弓大火変換':np.array([[7, 249, 248, 191, 166, 143, 46, 244]], dtype='uint8'),
    '槍大火変換':np.array([[7, 249, 248, 191, 166, 143, 46, 244]], dtype='uint8'),
    '騎大火変換':np.array([[7, 249, 248, 191, 166, 143, 46, 244]], dtype='uint8'),
    '術大火変換':np.array([[7, 249, 248, 191, 166, 143, 46, 244]], dtype='uint8'),
    '殺大火変換':np.array([[7, 249, 248, 191, 166, 143, 174, 244]], dtype='uint8'),
    '狂大火変換':np.array([[7, 249, 248, 191, 166, 143, 174, 244]], dtype='uint8'),
    '剣猛火変換':np.array([[11, 240, 248, 159, 38, 143, 206, 244]], dtype='uint8'),
    '弓猛火変換':np.array([[11, 240, 248, 157, 38, 143, 142, 244]], dtype='uint8'),
    '槍猛火変換':np.array([[11, 242, 248, 143, 6, 143, 142, 244]], dtype='uint8'),
    '騎猛火変換':np.array([[11, 114, 248, 143, 6, 175, 142, 240]], dtype='uint8'),
    '術猛火変換':np.array([[11, 114, 248, 143, 6, 175, 142, 240]], dtype='uint8'),
    '殺猛火変換':np.array([[11, 114, 248, 143, 6, 175, 142, 240]], dtype='uint8'),
    '狂猛火変換':np.array([[11, 122, 248, 143, 6, 175, 142, 240]], dtype='uint8'),
}

#種火のクラス見分けるハッシュ値
dist_tanebi_class = {
    '全種火':np.array([[224, 223, 56, 15, 62, 57, 7, 5]], dtype='uint8'),
    '剣種火':np.array([[217, 95, 165, 53, 62, 39, 14, 8]], dtype='uint8'),
    '弓種火':np.array([[118, 63, 137, 53, 44, 43, 42, 5]], dtype='uint8'),
    '槍種火':np.array([[226, 51, 37, 45, 120, 49, 42, 64]], dtype='uint8'),
    '騎種火':np.array([[153, 111, 22, 22, 166, 35, 58, 8]], dtype='uint8'),
    '術種火':np.array([[84, 157, 39, 35, 60, 87, 26, 13]], dtype='uint8'),
    '殺種火':np.array([[249, 54, 22, 57, 186, 41, 38, 10]], dtype='uint8'),
    '狂種火':np.array([[73, 39, 246, 30, 58, 25, 55, 33]], dtype='uint8'),
    '剣灯火':np.array([[217, 95, 167, 33, 62, 39, 14, 10]], dtype='uint8'),
    '弓灯火':np.array([[118, 63, 137, 53, 44, 43, 42, 5]], dtype='uint8'),
    '槍灯火':np.array([[98, 179, 37, 45, 56, 51, 42, 73]], dtype='uint8'),
    '騎灯火':np.array([[153, 111, 23, 22, 166, 35, 58, 8]], dtype='uint8'),
    '術灯火':np.array([[92, 157, 39, 35, 60, 87, 58, 13]], dtype='uint8'),
    '殺灯火':np.array([[249, 54, 22, 57, 184, 41, 38, 10]], dtype='uint8'),
    '狂灯火':np.array([[73, 167, 246, 30, 58, 25, 247, 33]], dtype='uint8'),
    '剣大火':np.array([[216, 91, 229, 49, 58, 39, 30, 10]], dtype='uint8'),
    '弓大火':np.array([[119, 63, 137, 63, 173, 42, 42, 5]], dtype='uint8'),
    '槍大火':np.array([[227, 51, 37, 44, 120, 51, 43, 64]], dtype='uint8'),
    '騎大火':np.array([[185, 111, 54, 62, 238, 43, 58, 8]], dtype='uint8'),
    '術大火':np.array([[157, 157, 38, 59, 60, 86, 26, 141]], dtype='uint8'),
    '殺大火':np.array([[249, 38, 22, 57, 184, 41, 55, 10]], dtype='uint8'),
    '狂大火':np.array([[73, 39, 246, 62, 58, 27, 127, 33]], dtype='uint8'),
    '剣猛火':np.array([[216, 95, 245, 53, 62, 47, 30, 10]], dtype='uint8'),
    '弓猛火':np.array([[115, 63, 152, 63, 125, 42, 42, 7]], dtype='uint8'),
    '槍猛火':np.array([[225, 35, 20, 44, 120, 62, 59, 72]], dtype='uint8'),
    '騎猛火':np.array([[217, 111, 54, 22, 238, 35, 63, 8]], dtype='uint8'),
    '術猛火':np.array([[92, 221, 38, 35, 60, 86, 26, 9]], dtype='uint8'),
    '殺猛火':np.array([[249, 55, 22, 25, 252, 43, 39, 202]], dtype='uint8'),
    '狂猛火':np.array([[89, 39, 246, 94, 58, 27, 23, 161]], dtype='uint8'),
    '剣業火':np.array([[153, 219, 133, 41, 126, 43, 63, 26]], dtype='uint8'),
    '弓業火':np.array([[115, 175, 24, 42, 45, 42, 106, 4]], dtype='uint8'),
    '槍業火':np.array([[65, 163, 5, 44, 120, 58, 127, 88]], dtype='uint8'),
##    '騎業火':np.array(None, dtype='uint8'),
    '術業火':np.array([[17, 189, 102, 106, 60, 22, 50, 9]], dtype='uint8'),
    '殺業火':np.array([[233, 230, 22, 57, 230, 59, 39, 158]], dtype='uint8'),
    '狂業火':np.array([[17, 103, 230, 122, 58, 27, 127, 32]], dtype='uint8'),
    '全種火変換':np.array([[224, 223, 56, 15, 62, 57, 7, 5]], dtype='uint8'),
    '剣種火変換':np.array([[217, 95, 165, 53, 62, 39, 14, 8]], dtype='uint8'),
    '弓種火変換':np.array([[118, 63, 137, 53, 44, 43, 42, 5]], dtype='uint8'),
    '槍種火変換':np.array([[226, 51, 37, 45, 120, 49, 42, 64]], dtype='uint8'),
    '騎種火変換':np.array([[153, 111, 22, 22, 166, 35, 58, 8]], dtype='uint8'),
    '術種火変換':np.array([[84, 157, 39, 35, 60, 87, 26, 13]], dtype='uint8'),
    '殺種火変換':np.array([[249, 54, 22, 57, 186, 41, 38, 10]], dtype='uint8'),
    '狂種火変換':np.array([[73, 39, 246, 30, 58, 25, 55, 33]], dtype='uint8'),
    '剣灯火変換':np.array([[217, 95, 167, 33, 62, 39, 14, 10]], dtype='uint8'),
    '弓灯火変換':np.array([[118, 63, 137, 53, 44, 43, 42, 5]], dtype='uint8'),
    '槍灯火変換':np.array([[98, 179, 37, 45, 56, 51, 42, 73]], dtype='uint8'),
    '騎灯火変換':np.array([[153, 111, 23, 22, 166, 35, 58, 8]], dtype='uint8'),
    '術灯火変換':np.array([[92, 157, 39, 35, 60, 87, 58, 13]], dtype='uint8'),
    '殺灯火変換':np.array([[249, 54, 22, 57, 184, 41, 38, 10]], dtype='uint8'),
    '狂灯火変換':np.array([[73, 167, 246, 30, 58, 25, 247, 33]], dtype='uint8'),
    '剣大火変換':np.array([[216, 91, 229, 49, 58, 39, 30, 10]], dtype='uint8'),
    '弓大火変換':np.array([[119, 63, 137, 63, 173, 42, 42, 5]], dtype='uint8'),
    '槍大火変換':np.array([[227, 51, 37, 44, 120, 51, 43, 64]], dtype='uint8'),
    '騎大火変換':np.array([[185, 111, 54, 62, 238, 43, 58, 8]], dtype='uint8'),
    '術大火変換':np.array([[157, 157, 38, 59, 60, 86, 26, 141]], dtype='uint8'),
    '殺大火変換':np.array([[249, 38, 22, 57, 184, 41, 55, 10]], dtype='uint8'),
    '狂大火変換':np.array([[73, 39, 246, 62, 58, 27, 127, 33]], dtype='uint8'),
    '剣猛火変換':np.array([[216, 95, 245, 53, 62, 47, 30, 10]], dtype='uint8'),
    '弓猛火変換':np.array([[113, 63, 153, 63, 61, 43, 42, 7]], dtype='uint8'),
    '槍猛火変換':np.array([[97, 55, 21, 12, 120, 51, 42, 64]], dtype='uint8'),
    '騎猛火変換':np.array([[217, 111, 54, 22, 238, 35, 63, 8]], dtype='uint8'),
    '術猛火変換':np.array([[92, 221, 38, 35, 60, 86, 26, 9]], dtype='uint8'),
    '殺猛火変換':np.array([[249, 55, 22, 25, 252, 43, 39, 202]], dtype='uint8'),
    '狂猛火変換':np.array([[89, 39, 246, 94, 58, 27, 23, 161]], dtype='uint8'),
}

dist_local = {
}

#恒常アイテム
#ここに記述したものはドロップ数を読み込まない
#順番ルールにも使われる
std_item = [ '爪', '心臓', '逆鱗', '根', '幼角', '涙石', '脂', 'ランプ',
    'スカラベ', '産毛', '胆石', '神酒', '炉心', '鏡', '卵', 'カケラ',
    '種', 'ランタン', '八連', '宝玉', '羽根', '歯車', '頁', 'ホム',
    '蹄鉄', '勲章', '貝殻', '勾玉', '結氷', '指輪', 'オーロラ', '鈴',
    '矢尻', '冠',
    '証', '骨', '牙', '塵', '鎖', '毒針', '髄液', '鉄杭', '火薬',
    '剣秘', '弓秘', '槍秘', '騎秘', '術秘', '殺秘', '狂秘',
    '剣魔', '弓魔', '槍魔', '騎魔', '術魔', '殺魔', '狂魔',
    '剣輝', '弓輝', '槍輝', '騎輝', '術輝', '殺輝', '狂輝',
    '剣モ', '弓モ', '槍モ', '騎モ', '術モ', '殺モ', '狂モ',
    '剣ピ', '弓ピ', '槍ピ', '騎ピ', '術ピ', '殺ピ', '狂ピ',
    '全種火', '全灯火', '全大火', '"全猛火','全業火',
    '剣種火', '剣灯火', '剣大火', '剣猛火', '剣業火',
    '弓種火', '弓灯火', '弓大火', '弓猛火', '弓業火',
    '槍種火', '槍灯火', '槍大火', '槍猛火', '槍業火',
    '騎種火', '騎灯火', '騎大火', '騎猛火', '騎業火',
    '術種火', '術灯火', '術大火', '術猛火', '術業火',
    '殺種火', '殺灯火', '殺大火', '殺猛火', '殺業火',
    '狂種火', '狂灯火', '狂大火', '狂猛火', '狂業火',
]

dist_card = {
    '礼装SSR':np.array([[ 11,  11, 173, 226, 182,  79, 179, 178]], dtype='uint8'),
    '礼装SR':np.array([[ 39, 128, 203, 233, 184, 185, 118,  78]], dtype='uint8'),
##    '礼装SR':np.array([[7, 129, 203, 233, 184, 185, 102, 111]], dtype='uint8'),
    
##    '礼装EXP':np.array([[7, 129, 203, 233, 184, 185, 102,  111]], dtype='uint8'),
##    '礼装SSR':np.array([[201, 137,  59, 238,  50,  90,  44, 230]], dtype='uint8'),
##    '礼装SSR2':np.array([[ 11,  11, 173, 226, 182,  79, 179, 178]], dtype='uint8'),
##    '礼装SSR3':np.array([[ 11,   9, 173, 226, 180, 238, 177, 182]], dtype='uint8'),

    'Point01':np.array([[ 15, 169, 234, 150, 168, 123, 194,  59]], dtype='uint8'),
    'Point02':np.array([[143,  41, 194, 167,  60, 219,  44, 150]], dtype='uint8'),
## 9.0 [[135 169 234 215 184 103 214 239]]

##    'Point01':np.array([[143,  41, 194, 167,  60, 219,  44, 150]], dtype='uint8'),
##    'Point02':np.array([[ 15, 169, 234, 150, 168, 123, 194, 187]], dtype='uint8'),
##    'Point03':np.array([[ 47, 232, 226, 167, 188, 251, 236, 182]], dtype='uint8'),
##    'Point04':np.array([[ 15, 169, 226, 167, 188, 211, 172, 182]], dtype='uint8'),
##    'Point05':np.array([[ 15, 169, 226, 134, 184, 123, 162,  62]], dtype='uint8'),
##    'Point06':np.array([[ 31, 169, 226, 134, 168, 121, 198,  59]], dtype='uint8'),
##    'Point07':np.array([[  7, 169, 234, 150, 168,  62, 198, 239]], dtype='uint8'),
##    'Point07':np.array([[135, 169, 234, 150, 168, 127, 246, 201]], dtype='uint8'),
}
    
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
    """
    return max(a[0], b[0]) <= min(a[2], b[2]) \
           and max(a[1], b[1]) <= min(a[3], b[3])


class ScreenShot:
    def __init__(self, img_rgb, svm, svm_chest, svm_card):
        threshold = 80

        self.img_rgb = img_rgb
        self.img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        th, self.img_th = cv2.threshold(self.img_gray, threshold, 255, cv2.THRESH_BINARY)
        self.svm = svm
        self.svm_chest = svm_chest
        
        self.height, self.width = img_rgb.shape[:2]
        self.pagenum, self.pages, self.lines = pageinfo.guess_pageinfo(img_rgb)
        self.chestnum = self.ocr_tresurechest()
        item_pts = []
        if self.chestnum >= 0:
            item_pts = self.img2points()

        self.items = []
        for i, pt in enumerate(item_pts):
            item_img_rgb = self.img_rgb[pt[1] :  pt[3],  pt[0] :  pt[2]]
            item_img_gray = self.img_gray[pt[1] :  pt[3],  pt[0] :  pt[2]]
            if i >= 14:
                self.items.append(Item(item_img_rgb, item_img_gray, svm, svm_card, bottom=True))
            else:
                self.items.append(Item(item_img_rgb, item_img_gray, svm, svm_card))
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
##            elif self.pagenum != 1:
##                itemlist.append(name + item.dropnum)                
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
            if item.name[-1].isdigit():
                name = item.name + '_'
            else:
                name = item.name
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
        pt = self.tresurechest_pt()
        img_num = self.img_th[pt[1]:pt[3],pt[0]:pt[2]]
        im_th = cv2.bitwise_not(img_num)
        h, w = im_th.shape[:2]

        #情報ウィンドウが数字とかぶった部分を除去する
        for y in range(h):
            im_th[y, 0] = 255
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


    def tresurechest_pt(self):
        """
        OCRで宝箱数を読む領域 [left, top, right, bottom]
        解像度別に設定
        左はぎりぎりまで切り詰めないと間に文字が有ると誤認識する
        """
        if self.width == 2560 and self.height == 1600:
            pt = [1989, 95, 2065, 153]
        elif self.width == 2436 and self.height == 1125:
            pt = [1740, 19, 1799, 60]
        elif self.width == 2048 and self.height == 1536:
##            pt = [1593, 211, 1660, 255]
            pt = [1592, 211, 1660, 255]
        elif self.width == 2048 and self.height == 1152:
            pt = [1592, 21, 1650, 59]
        elif self.width == 2048 and self.height == 877:
            pt = [1405, 16, 1455, 48]
        elif self.width == 1920 and self.height == 1200:
            pt = [1492, 77, 1560, 113]
        elif self.width == 1920 and self.height == 1080:
            pt = [1492, 16, 1533, 57]
        elif self.width == 1792 and self.height == 828:
            pt = [1282, 12, 1337, 40]
        elif self.width == 1334 and self.height == 750:
            pt = [1036, 14, 1073, 39]
        elif self.width == 2224 and self.height == 1668:
            pt = [1730, 232, 1785, 270]
        else:
            pt = []
        return pt

    def calc_offset(self, pts, std_pts, margin_x):
        """
        オフセットを反映
        """
        ## Y列でソート
        pts.sort(key=lambda x: x[1])

        ## Offsetを算出
        offset_x = pts[0][0] -margin_x 
        offset_y = pts[0][1] - std_pts[0][1]
        if offset_y > 30: #これ以上になったら二行目の座標と判断
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
        elif self.pagenum >= 2:
            item_pts = item_pts[14-(self.lines+2)%3*7:15+self.chestnum%7]
            
        return item_pts

    def booty_pts(self):
        """
        戦利品が出現する21の座標 [left, top, right, bottom]
        解像度別に設定
        """
        if self.width == 2560 and self.height == 1600:
            pts = [(309, 330, 546, 587), (584, 329, 821, 587), (859, 329, 1096, 587), (1134, 329, 1371, 587), (1409, 329, 1646, 587), (1684, 329, 1921, 587), (1959, 329, 2196, 587),
                       (309, 614, 546, 871), (584, 613, 821, 871), (859, 613, 1096, 871), (1134, 613, 1371, 871), (1409, 613, 1646, 871), (1684, 613, 1921, 871), (1959, 613, 2196, 871),
                       (309, 898, 546, 1155), (584, 898, 821, 1155), (859, 898, 1096, 1155), (1134, 898, 1371, 1155), (1409, 898, 1646, 1155), (1684, 898, 1921, 1155), (1959, 898, 2196, 1155)]
        elif self.width == 2436 and self.height == 1125:
            pts = [(502, 184, 676, 375), (705, 184, 879, 375), (908, 184, 1082, 375), (1111, 184, 1285, 375), (1313, 184, 1487, 375), (1516, 184, 1690, 375), (1719, 184, 1893, 375),
                       (502, 393, 676, 584), (705, 393, 879, 584), (908, 393, 1082, 584), (1111, 393, 1285, 584), (1313, 393, 1487, 584), (1516, 393, 1690, 584), (1719, 393, 1893, 584),
                       (502, 602, 676, 793), (705, 602, 879, 793), (908, 602, 1082, 793), (1111, 602, 1285, 793), (1313, 602, 1487, 793), (1516, 602, 1690, 793), (1719, 602, 1893, 793)]        
        elif self.width == 2048 and self.height == 1536:
            pts = [(247, 385, 437, 593), (467, 385, 657, 593), (687, 385, 877, 593), (907, 385, 1097, 593), (1127, 385, 1317, 593), (1347, 385, 1537, 593), (1567, 385, 1757, 593),
                       (247, 612, 437, 820), (467, 612, 657, 820), (687, 612, 877, 820), (907, 612, 1097, 820), (1127, 612, 1317, 820), (1347, 612, 1537, 820), (1567, 612, 1757, 820),
                       (247, 839, 437, 1047), (467, 839, 657, 1047), (687, 839, 877, 1047), (907, 839, 1097, 1047), (1127, 839, 1317, 1047), (1347, 839, 1537, 1047), (1567, 839, 1757, 1047)]
        elif self.width == 1920 and self.height == 1200:
            pts = [(232, 247, 409, 441), (438, 247, 615, 441), (644, 247, 821, 441), (851, 247, 1028, 441), (1057, 247, 1234, 441), (1263, 247, 1440, 441), (1470, 247, 1647, 441),
                       (232, 460, 409, 654), (438, 460, 615, 654), (644, 460, 821, 654), (851, 460, 1028, 654), (1057, 460, 1234, 654), (1263, 460, 1440, 654), (1470, 460, 1647, 654),
                       (232, 673, 409, 867), (438, 673, 615, 867), (644, 673, 821, 867), (851, 673, 1028, 867), (1057, 673, 1234, 867), (1263, 673, 1440, 867), (1470, 673, 1647, 867)]
        elif self.width == 1334 and self.height == 750:
            pts = [(168, 137, 292, 272), (311, 137, 435, 272), (455, 137, 579, 272), (598, 137, 722, 272), (741, 137, 865, 272), (884, 137, 1008, 272), (1028, 137, 1152, 272),
                       (168, 285, 292, 420), (311, 285, 435, 420), (455, 285, 579, 420), (598, 285, 722, 420), (741, 285, 865, 420), (884, 285, 1008, 420), (1028, 285, 1152, 420),
                       (168, 433, 292, 568), (311, 433, 435, 568), (455, 433, 579, 568), (598, 433, 722, 568), (741, 433, 865, 568), (884, 433, 1008, 568), (1028, 433, 1152, 568)]
        elif self.width == 2224 and self.height == 1668:
            criteria_left = 269
            criteria_top = 425
            item_width = 204
            item_height = 223
            margin_width = 35
            margin_height = 24
            pts = generate_booty_pts(criteria_left, criteria_top,
                item_width, item_height, margin_width, margin_height)
        else:
            pts = []

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
    def __init__(self, img_rgb, img_gray, svm, svm_card, bottom=False):
        self.img_rgb = img_rgb
        self.img_gray = img_gray
        self.img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)
        self.height, self.width = img_rgb.shape[:2]
        self.card = self.classify_card(svm_card)
        self.name = self.classify_item(img_rgb)
        self.svm = svm
        if self.name not in std_item:
            self.dropnum = self.ocr_digit(bottom)
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
        img_hsv_lower = self.img_hsv[int(self.height*0.72):int(self.height*0.89), 8:self.width-8]
        h, w = img_hsv_lower.shape[:2]
        # 手持ちスクショでうまくいっている範囲
        # 黄文字がこの数値でマスクできるかが肝
        lower_yellow = np.array([25,180,119]) 
        upper_yellow = np.array([37,255,255])

        img_hsv_lower_mask = cv2.inRange(img_hsv_lower, lower_yellow, upper_yellow)
        contours = cv2.findContours(img_hsv_lower_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]

        item_pts_lower_yellow = []
        # 物体検出マスクがうまくいっているかが成功の全て
        for cnt in contours:
            ret = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            pt = [ ret[0], ret[1], ret[0] + ret[2], ret[1] + ret[3] ]
            if ret[2] < int(w/2) and ret[1] < int(h*3/5) and area > 1:
                item_pts_lower_yellow = self.conflictcheck(item_pts_lower_yellow, pt)

        item_pts_lower_yellow.sort()
        if len(item_pts_lower_yellow) > 0:
            if w - item_pts_lower_yellow[-1][2] > int((15*self.width/190)):
                #黄文字は必ず右寄せなので最後の文字が画面端から離れている場合全部ゴミ
                item_pts_lower_yellow = []

        return self.extension(item_pts_lower_yellow)


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


    def detect_lower_white_char(self, img_hsv_lower, im_th_lower):
        """
        戦利品数OCRで銀枠を除く(Reward QPは含む)下段の白文字の座標を抽出する

        ノイズは少し気になる程度
        """
        h, w = img_hsv_lower.shape[:2]
        digitimg = self.img2digitimg(img_hsv_lower, im_th_lower)
        contours = cv2.findContours(digitimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

        item_pts_lower_white = []
        new_contours = []
        for cnt in contours:
            ret = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            pt = [ ret[0], ret[1], ret[0] + ret[2], ret[1] + ret[3] ]
            if ret[1] + ret[3] > int(h/2) and ret[1] < int(h/2) and area > 1:
                item_pts_lower_white = self.conflictcheck(item_pts_lower_white, pt)

        item_pts_lower_white.sort()

        return self.extension_straighten(item_pts_lower_white)

    def detect_lower_white_char4silver(self, im_th_lower,item_pts_lower_yellow, bottom):
        """
        戦利品数OCRで銀枠の下段白文字の座標を抽出する

        ※銀枠のみ白でマスクしてもうまく動かないため別処理
        二値化画像でオブジェクト検出
        →抽出したオブジェクトのうち文字であると推定される部分以外の座標を白にする
        →反転、文字のみ白の画像ができる
        →オブジェクト検出
        ノイズ多め、おかしなことが起こる可能性が最も高い処理
        二値化に閾値によっては同じ画像の隣り合う二つの同一アイテムで片方は成功、
        片方は失敗というケースもある

        """
        h, w = im_th_lower.shape[:2]

        contours = cv2.findContours(im_th_lower, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

        #オブジェクト検出(1回目)
        new_contours = []
        for cnt in contours:
            ret = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            pt = [ ret[0], ret[1], ret[0] + ret[2], ret[1] + ret[3] ]
            if ret[1] + ret[3] > int(h/2) and ret[1] < int(h/2) and area > 1:
                new_contours.append(cnt)

        # 文字であると推定される部分以外の座標を白に
        im_th_lower2 = im_th_lower.copy()
        for y in range(h):
            for x in range(w):
                innerflag = False
                for cnt in new_contours:
                    if cv2.pointPolygonTest(cnt, (x, y), 0) > 0:
                        innerflag = True
                if innerflag == False:
                    im_th_lower2[y, x] = 255
        im_th_lower_rev = cv2.cv2.bitwise_not(im_th_lower2)
        kernel1 = np.ones((3,1),np.uint8)
        dilation = cv2.dilate(im_th_lower_rev,kernel1,iterations = 1)

         #物体検出を成功させるために下端を黒に染める
        if bottom == True:
            for x in range(w):
                for y in range(int(7/23*h)):
                    dilation[h - y -1, x] = 0

        #オブジェクト検出(2回目)
        contours = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
        h, w = im_th_lower.shape[:2]
        item_pts_lower_white = []
        for cnt in contours:
            ret = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            if ret[2] < int(w/2) and ret[2] > 5 and area > 1 and ret[1] + ret[3] > int(h/2)  and ret[1] < int(h*5/3):
            ## ret[2]の幅制限をいれると + や 1 を認識しなくなる問題
    ##        if ret[2] < int(w/2) and area > 1  and ret[1] + ret[3] > int(h/2) and ret[1] < int(h*5/3):
                if bottom == True:
                    pt = [ ret[0], ret[1], ret[0] + ret[2], ret[1] + ret[3] + int(4/23*h)]
                else:
                    pt = [ ret[0], ret[1], ret[0] + ret[2], ret[1] + ret[3]]
                if len(item_pts_lower_yellow) > 0:
                    if ret[0] + ret[2] > item_pts_lower_yellow[0][0]:
                        continue
                item_pts_lower_white = self.conflictcheck(item_pts_lower_white, pt)

        item_pts_lower_white.sort()
        return self.extension_straighten(item_pts_lower_white)

    def detect_upper_char(self, img_hsv_upper, im_th_upper, img_rgb_upper):
        """
        戦利品数OCRで上段白文字の座標を抽出する(黄文字は上段に出現しない)

        銀枠の場合、背景のアイテムが重なる部分以外認識失敗しやすいのでエラー訂正でカバー

        """
        h, w = img_hsv_upper.shape[:2]
        digitimg = self.img2digitimg(img_hsv_upper, im_th_upper)

        #下2桁切り出して上下幅を決める
        img_right = digitimg[0:h, int(128/174*w):w]
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
            if area > 15:
                tmp_pts.append(pt)
        h_top = tmp_pts[-1][1]
        h_bottom = tmp_pts[-1][3]
        
        #物体検出を成功させるために右端を黒に染める
        for y in range(h):
            for x in range(int(7*161/w)):
                digitimg[y, w-x-1] = 0
         #物体検出を成功させるために上下端を黒に染める
        for x in range(w):
            for y in range(h_top):
                digitimg[y, x] = 0
        for x in range(w):
            for y in range(h - h_bottom):
                digitimg[h - y -1, x] = 0

        contours = cv2.findContours(digitimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

        item_pts_upper = []
        for cnt in contours:
            ret = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            pt = [ ret[0], ret[1], ret[0] + ret[2], ret[1] + ret[3] ]
##            if ret[1] + ret[3] > int(h/2) and ret[1] < int(h*0.7) and ret[2] < int(35/190*w) and area > 15 and ret[2] > int(9/190*w): #9/190でギリギリ
##                item_pts_upper = self.conflictcheck(item_pts_upper, pt)
            if ret[1] + ret[3] > int(h/2) and  ret[1] < int(h*0.7) and ret[2] < int(40/190*w) and ret[1] < int(h*0.7) and area > 15 and ret[2] > int(9/190*w):
##                and ret[1] < int(h*0.7) and ret[2] < int(35/190*w) and area > 15 and ret[2] > int(9/190*w): #9/190でギリギリ
##                print(ret[2])
                item_pts_upper = self.conflictcheck(item_pts_upper, pt)

        item_pts_upper.sort()

        return self.extension_straighten(item_pts_upper)

    def read_item(self, img_gray, pts, upper=False, yellow=False,):
        """
        戦利品の数値をOCRする(エラー訂正有)
        """
        width = img_gray.shape[1]

        win_size = (120, 60)
        block_size = (16, 16)
        block_stride = (4, 4)
        cell_size = (4, 4)
        bins = 9
        lines = ""

        for pt in pts:
            char = []
            tmpimg = img_gray[pt[1]:pt[3], pt[0]:pt[2]]
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
                for i in range(int((width-pts[-1][2])/(21/190*width))):
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

    def ocr_digit(self, bottom):
        """
        戦利品OCR
        bottom はアイテム出現部が最下部かどうか
        """
        ## 50が0.0.2
    ##    th, im_th = cv2.threshold(img_gray, 176, 255, cv2.THRESH_BINARY)
        th, im_th = cv2.threshold(self.img_gray, 174, 255, cv2.THRESH_BINARY)
        #174じゃないとうまくいかない IMG_8666
        #170より大きくすると0が()になる場合がある(のちにエラー訂正有)
        #176にしないとうまく分割できないときがある
        im_th2 = cv2.cv2.bitwise_not(im_th)

        #この数値は固定で良い
        img_rgb_top = self.img_rgb[24:34, 7:17]
        img_hsv_top = self.img_hsv[24:34, 7:17]

       
        ### QPが7桁とかになるとベースラインが下がる?
        im_th_upper = im_th2[int(self.height*0.575):int(self.height*0.745), 8:self.width-8]
        img_rgb_upper = self.img_rgb[int(self.height*0.575):int(self.height*0.745), 8:self.width-8]
        img_hsv_upper = self.img_hsv[int(self.height*0.575):int(self.height*0.745), 8:self.width-8]
        img_gray_upper = self.img_gray[int(self.height*0.575):int(self.height*0.745), 8:self.width-8]

        im_th_lower = im_th2[int(self.height*0.72):int(self.height*0.89), 8:self.width-8]
        img_hsv_lower = self.img_hsv[int(self.height*0.72):int(self.height*0.89), 8:self.width-8]
        img_gray_lower = self.img_gray[int(self.height*0.72):int(self.height*0.89), 8:self.width-8]

        flag_silver = False
        if self.is_silver_item() == True:
            flag_silver = True
            
        item_pts_lower_yellow = self.detect_lower_yellow_char()
        line_lower_yellow = self.read_item(img_gray_lower, item_pts_lower_yellow, yellow=True)

        item_pts_lower_white = []
        line_upper = ""
        line_lower_white = ""
        upper_flag = False
        # +24を超えると下段にデータは無いと判断
        if len(line_lower_yellow) >= 5:
            if self.name == "QP":
                upper_flag = True            
            elif int(line_lower_yellow[2:-1]) > 24:
                upper_flag = True
        if upper_flag == True:
            item_pts_upper = self.detect_upper_char(img_hsv_upper, im_th_upper, img_rgb_upper)
            line_upper = self.read_item(img_gray_upper, item_pts_upper, upper=True)
        else:
            if self.card == "Point" or flag_silver == True:
                item_pts_lower_white = self.detect_lower_white_char4silver(im_th_lower, item_pts_lower_yellow, bottom)
            else:
                item_pts_lower_white = self.detect_lower_white_char(img_hsv_lower, im_th_lower)

            line_lower_white = self.read_item(img_gray_lower, item_pts_lower_white)

        drop = line_upper + line_lower_white+line_lower_yellow
        drop =re.sub("\([^\(\)]*\)$", "", drop) #括弧除去
        if drop != "":
            drop = "(" + drop + ")"
        return drop

    def classify_standard_item(self, img):
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
            if item[0].endswith("魔"):
                hash_ma = self.compute_maseki_hash(img)
                masekifiles = {}
                for i in dist_maseki.keys():
                    d2 = hasher.compare(hash_ma, dist_maseki[i])
                    if d2 <= 20:
                        masekifiles[i] = d2
                masekifiles = sorted(masekifiles.items(), key=lambda x:x[1])
                item = next(iter(masekifiles))
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
        carddic = { 0:'Quest Reward', 1:'Item', 2:'Point', 3:'Craft Essence', 99:"" }

        tmpimg = self.img_rgb[int(188/206*self.height):int(200/206*self.height),
                      int(77/188*self.width):int(114/188*self.width)]
        tmpimg = cv2.resize(tmpimg, (win_size))
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, bins)
        test.append(hog.compute(tmpimg)) # 特徴量の格納
        test = np.array(test)
        pred = svm_card.predict(test)

        return carddic[pred[1][0][0]]
        
    def classify_item(self, img):
        """
        アイテム判別器
        """
        if self.card == "Point":
            return "ポイント"
        item = self.classify_standard_item(img)
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


def get_output(filenames):
    """
    出力内容を作成
    """
    calc_dist_local()
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
        f = Path(filename)

        if f.exists() == False:
            output = { 'filename': filename + ': Not Found' }
        else:            
            img_rgb = imread(filename)

            try:
                sc = ScreenShot(img_rgb, svm, svm_chest, svm_card)

                #2頁目以降のスクショが無い場合に migging と出力                
                if prev_pages - prev_pagenum > 0 and sc.pagenum == 1:
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
                if sc.chestnum >= 21 and sc.lines >= 4 and sc.pagenum == 1 \
                   or sc.chestnum >= 42 and sc.lines >= 7 and sc.pagenum == 2:
                    output["ドロ数"] = str(output["ドロ数"]) + "+"
                output.update(sc.allitemdic)
            except:
                output = ({'filename': filename + ': not valid'})
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
    parser.add_argument('--version', action='version', version=progname + " " + version)

    args = parser.parse_args()    # 引数を解析

    if not Item_dir.is_dir():
        Item_dir.mkdir()

    csvfieldnames, outputcsv = get_output(args.filenames)
    
    fnames = csvfieldnames.keys()
    writer = csv.DictWriter(sys.stdout, fieldnames=fnames, lineterminator='\n')
    writer.writeheader()
    if len(outputcsv) > 1: #ファイル一つのときは合計値は出さない
        writer.writerow(csvfieldnames)
    for o in outputcsv:
        writer.writerow(o)
