#!/usr/bin/env python3
## スクショ集計したCSVから周回カウンタ書式に変換
import csv
import sys
import argparse
from pathlib import Path
import json
import re
import fgosccnt

drop_file = Path(__file__).resolve().parent / Path("hash_drop.json")
with open(drop_file, encoding='UTF-8') as f:
    drop_item = json.load(f)

shortname2id = {item["shortname"]:item["id"] for item in drop_item if "shortname" in item.keys()}
name2id = {item["name"]:item["id"] for item in drop_item if "name" in item.keys()}
fgosccnt.calc_dist_local()

ID_GEM_MIN = 6001
ID_GEM_MAX = 6007
ID_MAGIC_GEM_MIN = 6101
ID_MAGIC_GEM_MAX = 6107
ID_SECRET_GEM_MIN = 6201
ID_SECRET_GEM_MAX = 6207
ID_PIECE_MIN = 7001
ID_MONUMENT_MAX = 7107
ID_STAMDARD_ITEM_MIN = 6501
ID_STAMDARD_ITEM_MAX = 6599

def delete_brackets(s):
    """
    括弧と括弧内文字列を削除
    """
    """ brackets to zenkaku """
    table = {
        "(": "（",
        ")": "）",
##        "<": "＜",
##        ">": "＞",
##        "{": "｛",
##        "}": "｝",
##        "[": "［",
##        "]": "］"
    }
    for key in table.keys():
        s = s.replace(key, table[key])
    """ delete zenkaku_brackets """
##    l = ['（[^（|^）]*）', '【[^【|^】]*】', '＜[^＜|^＞]*＞', '［[^［|^］]*］',
##         '「[^「|^」]*」', '｛[^｛|^｝]*｝', '〔[^〔|^〕]*〕', '〈[^〈|^〉]*〉']
    l = ['（[^（|^）]*）']
    for l_ in l:
        s = re.sub(l_, "", s)
    """ recursive processing """
    return delete_brackets(s) if sum([1 if re.search(l_, s) else 0 for l_ in l]) > 0 else s

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('infile', nargs='?', type=argparse.FileType(),
                        default=sys.stdin)
    parser.add_argument('--place')
    parser.add_argument('--point', default='ポイント', help="'Point' item's name")
    args = parser.parse_args()

    with args.infile as f:
        reader = csv.DictReader(f)
        l = [row for row in reader]

    warning = ""
    for i, item in enumerate(l):
        if item['filename'] == "missing":
            warning = warning + "{}行目に missing なデータがあります\n".format(i+2)
##            print("missing なデータがあります", file=sys.stderr)
##            sys.exit(1)
        elif item['filename'].endswith("not valid"):
            warning = warning + "{}行目に not valid なデータがあります\n".format(i+2)
        elif item['filename'].endswith("not found"):
            warning = warning + "{}行目に not found なデータがあります\n".format(i+2)

    if warning != "":
        print ("""###############################################
# WARNING: この処理には以下のエラーがあります #
#      結果をそのまま使用しないでください     #
###############################################
{}###############################################""".format(warning))

    place = ""
    if l[0]["filename"] != "合計":
        place = l[0]["filename"]
    else:
        place = "周回場所"    
    if args.place:
        place = args.place
    print ("【{}】".format(place), end="")
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
            if delete_brackets(item) in shortname2id.keys():
                id = shortname2id[delete_brackets(item)]
            elif delete_brackets(item) in name2id.keys():
                id = name2id[delete_brackets(item)]
            else:
                id = [k for k, v in fgosccnt.item_name.items() if v == delete_brackets(item)][0]
            if stditem_flag == False \
               and ID_STAMDARD_ITEM_MIN <= id <=  ID_STAMDARD_ITEM_MAX:
                output = output[:-1] + "\n"
                stditem_flag = True
            elif stditem_flag == True \
                 and not (ID_STAMDARD_ITEM_MIN <= id <=  ID_STAMDARD_ITEM_MAX):
                output = output[:-1] + "\n"
                stditem_flag = False

            if skillstone_flag == False \
               and ID_GEM_MIN <= id <= ID_GEM_MAX:
                output = output[:-1] + "\n"
                skillstone_flag = True
            elif skillstone_flag == True \
                 and not (ID_GEM_MIN <= id <= ID_GEM_MAX):
                output = output[:-1] + "\n"
                skillstone_flag = False

            if monyupi_flag == False \
               and ID_PIECE_MIN <= id <= ID_MONUMENT_MAX:
                output = output[:-1] + "\n"
                monyupi_flag = True
            elif monyupi_flag == True \
                 and not (ID_PIECE_MIN <= id <= ID_MONUMENT_MAX):
                output = output[:-1] + "\n"
                monyupi_flag = False
            type = fgosccnt.item_type[id]
            if type == "Point" and point_flag == False:
                output = output[:-1] + "\n"
                point_flag = True
            elif point_flag == True and not type == "Point":
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
