#!/usr/bin/env python3
## スクショ集計したCSVから周回カウンタ書式に変換
import csv
import sys
import argparse

monyupi_list = ['剣モ', '弓モ', '槍モ', '騎モ', '術モ', '殺モ', '狂モ',
                '剣ピ', '弓ピ', '槍ピ', '騎ピ', '術ピ', '殺ピ', '狂ピ', ]

skillstone_list = [ '剣秘', '弓秘', '槍秘', '騎秘', '術秘', '殺秘', '狂秘',
                    '剣魔', '弓魔', '槍魔', '騎魔', '術魔', '殺魔', '狂魔',
                    '剣輝', '弓輝', '槍輝', '騎輝', '術輝', '殺輝', '狂輝']

stditem_list = ['爪',  '心臓',  '逆鱗',  '根',  '幼角',
                '涙石',  '脂',  'ランプ',  'スカラベ',  '産毛',
                '胆石',  '神酒',  '炉心',  '鏡',  '卵',  'カケラ',
                '種',  'ランタン',  '八連',  '宝玉',  '羽根',
                '歯車',  '頁',  'ホム',  '蹄鉄',  '勲章',
                '貝殻',  '勾玉',  '結氷',  '指輪',  'オーロラ',
                '鈴',  '矢尻',  '冠',
                '証',  '骨',  '牙',  '塵',  '鎖',
                '毒針',  '髄液',  '鉄杭',  '火薬']

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('infile', nargs='?', type=argparse.FileType(),
                        default=sys.stdin)
    parser.add_argument('--place', default='周回場所')
    parser.add_argument('--point', default='ポイント', help="'Point' item's name")
    args = parser.parse_args()

    with args.infile as f:
        reader = csv.DictReader(f)
        l = [row for row in reader]

    for item in l:
        if item['filename'] == "missing":
            print("missing なデータがあります", file=sys.stderr)
            sys.exit(1)
    print ("【{}】".format(args.place), end="")
    output = ""
    monyupi_flag = False
    skillstone_flag = False
    stditem_flag = False
    point_flag  = False
    qp_flag = False
    for i, item in enumerate(l[0].keys()):
        if i == 2:
            output =  l[0][item] + "周\n"
        if i > 2:
            if stditem_flag == False and item in stditem_list:
                output = output[:-1] + "\n"
                stditem_flag = True
            elif stditem_flag == True and item not in stditem_list:
                output = output[:-1] + "\n"
                stditem_flag = False

            if skillstone_flag == False and item in skillstone_list:
                output = output[:-1] + "\n"
                skillstone_flag = True
            elif skillstone_flag == True and item not in skillstone_list:
                output = output[:-1] + "\n"
                skillstone_flag = False

            if monyupi_flag == False and item in monyupi_list:
                output = output[:-1] + "\n"
                monyupi_flag = True
            elif monyupi_flag == True and item not in monyupi_list:
                output = output[:-1] + "\n"
                monyupi_flag = False

            if item.startswith('ポイント(+') and point_flag == False:
                output = output[:-1] + "\n"
                point_flag = True
            elif point_flag == True and not item.startswith('ポイント(+'):
                output = output[:-1] + "\n"
                point_flag = False

            if item.startswith('QP(+') and qp_flag == False:
                output = output[:-1] + "\n"
                qp_flag = True
            elif qp_flag == True and not item.startswith('QP(+'):
                output = output[:-1] + "\n"
                qp_flag = False
            output =  output + item + l[0][item] + "-"
    output = output.replace('ポイント(+', args.point + '(+')
    print (output[:-1])
    print ("#FGO周回カウンタ http://aoshirobo.net/fatego/rc/")
