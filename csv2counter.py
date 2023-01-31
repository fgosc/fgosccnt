#!/usr/bin/env python3
# スクショ集計したCSVから周回カウンタ書式に変換
import csv
import sys
import argparse
from pathlib import Path
import json
import re
import logging

import fgosccnt
sys.stdin = open(sys.stdin.fileno(), 'r', encoding='utf_8_sig')

logger = logging.getLogger(__name__)

basedir = Path(__file__).resolve().parent
drop_file = basedir / Path("fgoscdata/hash_drop.json")
with open(drop_file, encoding='UTF-8') as f:
    drop_item = json.load(f)

shortname2id = {
                item["shortname"]: item["id"]
                for item in drop_item if "shortname" in item.keys()
                }
name2id = {
           item["name"]: item["id"]
           for item in drop_item if "name" in item.keys()
           }
id2type = {item["id"]: item["type"] for item in drop_item}
# 未対応アイテムのための修正
for i in range(20):
    id2type[9500000 + i] = "Item"
item_shortname = {item["id"]: item["shortname"] for item in drop_item
                  if "shortname" in item.keys()}
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
ID_FREEQUEST_MIN = 93000001
ID_FREEQUEST_MAX = 93099999
ID_SYUERNQUEST_MIN = 94006801
ID_SYURENQUEST_MAX = 94061640
ID_SYUERNQUEST_TMP_MIN = 94066101
ID_SYURENQUEST_TMP_MAX = 94066199
ROW_ITEM_START = 4

output = ""
ce_exp_list = []
ce_list = []


def delete_brackets(s):
    """
    括弧と括弧内文字列を削除
    """
    """ brackets to zenkaku """
    table = {
        "(": "（",
        ")": "）",
    }
    for key in table.keys():
        s = s.replace(key, table[key])
    """ delete zenkaku_brackets """
    l = ['（[^（|^）]*）']
    for l_ in l:
        s = re.sub(l_, "", s)
    """ recursive processing """
    return delete_brackets(s) if sum(
                                     [1 if re.search(l_, s) else 0 for l_ in l]
                                     ) > 0 else s


def output_warning(lines):
    warning = ""
    # 報酬QP数でエラーチェック
    reward_qp = sum(x.startswith("報酬QP(") for x in lines[0].keys())
    if reward_qp > 1:
        warning += f"少なくとも{reward_qp}つのクエストの結果が混在しています\n"
    for i, item in enumerate(lines):
        if item['filename'] == "missing":
            warning = warning + "{}行目に missing (複数ページの撮影抜け)があります\n".format(i+2)
        elif item['filename'].endswith("not valid"):
            warning = warning + "{}行目に not valid (認識エラー)があります\n".format(i+2)
        elif item['filename'].endswith("not found"):
            warning = warning + "{}行目に not found なスクショがあります\n".format(i+2)
        elif item['filename'].endswith("duplicate"):
            warning = warning + "{}行目に直前と重複したスクショがあります\n".format(i+2)

    if warning != "":
        print("""###############################################
# WARNING: この処理には以下のエラーがあります #
#　確認せず結果をそのまま使用しないでください #
###############################################
{}###############################################""".format(warning))


def place2id(place, freequest):
    """
    フリクエと修練場のidが変換できればよい
    """
    tmp = place.split(" ")
    if len(tmp) >= 2:
        chapter = tmp[0]
        name = tmp[1]
    else:
        return -1
    for fq in freequest:
        if chapter == fq["chapter"] and name == fq["name"]:
            # 北米以外のフリクエ
            return fq["id"]
        elif chapter == fq["chapter"] and name == fq["place"]:
            # 北米以外のフリクエ(同じ場所に二つクエストがある場合)
            # 修練場もここで判定
            return fq["id"]
        elif chapter == fq["place"] and name == fq["name"]:
            # 北米
            return fq["id"]
    return -1


def output_header(lines):
    global ce_list
    global ce_exp_list
    global output
    output_warning(lines)
    place = ""
    if lines[0]["filename"] != "合計" and len(lines) > 1:
        # fgosccnt がクエスト名判別に成功した
        eventquest_dir = basedir / Path("fgoscdata/data/json/")
        freequest = []
        eventfiles = eventquest_dir.glob('**/*.json')
        for eventfile in eventfiles:
            try:
                with open(eventfile, encoding='UTF-8') as f:
                    event = json.load(f)
                    freequest = freequest + event
            except (OSError, UnicodeEncodeError) as e:
                logger.exception(e)

        place = lines[0]["filename"]
        # 場所からドロップリストを決定
        drop = []
        if place == "宝物庫 初級":
            questid = 94061636
        elif  place == "宝物庫 中級":
            questid = 94061637
        elif  place == "宝物庫 上級":
            questid = 94061638
        elif  place == "宝物庫 超級":
            questid = 94061639
        elif  place == "宝物庫 極級":
            questid = 94061640
        else:
            questid = place2id(place, freequest)
        logger.debug("questid: %d", questid)

        if not (ID_FREEQUEST_MIN <= questid <= ID_FREEQUEST_MAX) \
           and not (ID_SYUERNQUEST_MIN <= questid <= ID_SYURENQUEST_MAX) \
           and not (ID_SYUERNQUEST_TMP_MIN <= questid <= ID_SYURENQUEST_TMP_MAX) :
           # 通常フリクエと修練場は除く
            logger.debug("フリクエでも修練場でもないクエスト")
            for fq in freequest:
                if "shortname" in fq.keys():
                    if place == fq["shortname"]:
                        drop = fq["drop"]
                        break
            if drop == []:
                logger.critical("dropの取得に失敗")
                exit()
            # CEリストの作成
            for equip in drop:
                if equip["type"] == "Craft Essence":
                    if equip["name"].startswith("概念礼装EXPカード："):
                        ce_exp_list.append(equip)
                    else:
                        ce_list.append(equip)
            logger.debug("ce_list: %s", ce_list)
            logger.debug("ce_exp_list: %s", ce_exp_list)
    else:
        if len(lines[1]) <= 3:
            print("認識できるファイル(.JPG, .JPEG, or .PNG)がありません")
            sys.exit(0)
        place = "周回場所"
    if args.place:
        place = args.place
    print("【{}】".format(place), end="")

    # 周回数出力
    for i, item in enumerate(lines[0].keys()):
        if i == ROW_ITEM_START:
            output = lines[0][item] + "周\n"
            break


def output_ce(lines):
    global output
    # 礼装出力
    if len(ce_list) > 0:
        ce_output = {item_shortname[k["id"]]: 0 for k in ce_list}
        logger.debug("ce_output: %s", ce_output)
        for i, item in enumerate(lines[0].keys()):
            if i > ROW_ITEM_START:
                if delete_brackets(item) in shortname2id.keys():
                    id = shortname2id[delete_brackets(item)]
                elif delete_brackets(item) in name2id.keys():
                    id = name2id[delete_brackets(item)]
                else:
                    id = [
                          k for k, v in fgosccnt.item_name.items()
                          if v == delete_brackets(item)
                          ][0]
                # logger.debug("i: %s", i)
                # logger.debug("item: %s", item)
                # logger.debug("id: %s", id)
                if item == "礼装" and lines[0][item] == '0':
                    # 全てのイベント限定概念礼装を0出力する
                    for ce in ce_list:
                        output = output + item_shortname[ce["id"]] + '0' + "-"
                    break
                # 礼装複数ドロップで一部のみドロップしているとき
                # 礼装じゃないアイテムがでてきたら終了
                if id2type[id] == "Craft Essence" and not item.endswith("EXP礼装"):
                    ce_output[item] = lines[0][item]
                else:
                    for ce in ce_output.keys():
                        output = output + ce + str(ce_output[ce]) + "-"
                    break
    else:
        for i, item in enumerate(lines[0].keys()):
            if i > ROW_ITEM_START:
                if item.endswith("礼装"):
                    output = output + item + lines[0][item] + "-"




def output_ce_exp(lines):
    global output
    # EXP礼装出力
    if len(ce_exp_list) > 0:
        ce_exp_output = {item_shortname[k["id"]]: 0 for k in ce_exp_list}
        logger.debug("ce_exp_output: %s", ce_exp_output)
        for i, item in enumerate(lines[0].keys()):
            if i > ROW_ITEM_START:
                if delete_brackets(item) in shortname2id.keys():
                    id = shortname2id[delete_brackets(item)]
                elif delete_brackets(item) in name2id.keys():
                    id = name2id[delete_brackets(item)]
                else:
                    id = [
                          k for k, v in fgosccnt.item_name.items()
                          if v == delete_brackets(item)
                          ][0]
                # logger.debug("i: %s", i)
                # logger.debug("item: %s", item)
                # logger.debug("id: %s", id)
                # 礼装複数ドロップで一部のみドロップしているとき
                # 礼装じゃないアイテムがでてきたら終了
                logger.debug(item)
                if id2type[id] == "Craft Essence" and not item.endswith("EXP礼装"):
                    continue
                elif id2type[id] == "Craft Essence" and item.endswith("EXP礼装"):
                    ce_exp_output[item] = lines[0][item]
                else:
                    for ce_exp in ce_exp_output.keys():
                        output = output + ce_exp + str(ce_exp_output[ce_exp]) + "-"
                    break


def output_item(lines):
    global output
    global output_items
    monyupi_flag = False
    skillstone_flag = False
    stditem_flag = False
    point_flag = False
    qp_flag = False
    output_items = []
    # 礼装以外のアイテム出力
    for i, item in enumerate(lines[0].keys()):
        if i > ROW_ITEM_START:
            if delete_brackets(item) in shortname2id.keys():
                id = shortname2id[delete_brackets(item)]
            elif delete_brackets(item) in name2id.keys():
                id = name2id[delete_brackets(item)]
            else:
                id = [
                      k for k, v in fgosccnt.item_name.items()
                      if v == delete_brackets(item)
                      ][0]
            # logger.debug("i: %s", i)
            # logger.debug("item: %s", item)
            # logger.debug("id: %s", id)
            if not item.startswith("item"):
                if id2type[id] == "Craft Essence":
                    continue
            # 改行出力ルーチン
            if stditem_flag is False \
               and ID_STAMDARD_ITEM_MIN <= id <= ID_STAMDARD_ITEM_MAX:
                output = output[:-1] + "\n"
                stditem_flag = True
            elif stditem_flag and not (ID_STAMDARD_ITEM_MIN <= id <= ID_STAMDARD_ITEM_MAX):
                output = output[:-1] + "\n"
                stditem_flag = False

            if skillstone_flag is False and ID_GEM_MIN <= id <= ID_SECRET_GEM_MAX:
                output = output[:-1] + "\n"
                skillstone_flag = True
            elif skillstone_flag and not (ID_GEM_MIN <= id <= ID_SECRET_GEM_MAX):
                output = output[:-1] + "\n"
                skillstone_flag = False

            if monyupi_flag is False and ID_PIECE_MIN <= id <= ID_MONUMENT_MAX:
                output = output[:-1] + "\n"
                monyupi_flag = True
            elif monyupi_flag and not (ID_PIECE_MIN <= id <= ID_MONUMENT_MAX):
                output = output[:-1] + "\n"
                monyupi_flag = False
            type = fgosccnt.item_type[id]
            if type == "Point" and point_flag is False:
                output = output[:-1] + "\n"
                point_flag = True
            elif point_flag and not type == "Point":
                output = output[:-1] + "\n"
                point_flag = False

            if item.startswith('QP(+') and qp_flag is False:
                output = output[:-1] + "\n"
                qp_flag = True
            elif qp_flag and not item.startswith('QP(+'):
                output = output[:-1] + "\n"
                qp_flag = False
            if lines[0][item] == "":
                item_num = "0"
            else:
                item_num = lines[0][item]
            if item[-1].isdigit():
                output = output + item + "_" + item_num + "-"
            else:
                output = output + item + item_num + "-"
                output_items.append(item)
    output = output.replace('ポイント(+', args.point + '(+')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('infile', nargs='?', type=argparse.FileType('r', encoding='utf_8_sig'),
                        default=sys.stdin)
    parser.add_argument('--place')
    parser.add_argument('--point', default='ポイント', help="'Point' item's name")
    parser.add_argument('-l', '--loglevel',
                        choices=('debug', 'info'), default='info')
    args = parser.parse_args()
    lformat = '[%(levelname)s] %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=lformat,
    )
    logger.setLevel(args.loglevel.upper())

    with args.infile as f:
        reader = csv.DictReader(f)
        lines = [row for row in reader]

    output_header(lines)
    output_ce(lines)
    output_ce_exp(lines)
    output_item(lines)

    print(output[:-1])
    print("#FGO周回カウンタ http://aoshirobo.net/fatego/rc/")
    additional_list1 = ["最中", "団子", "煎餅",  # 鎌倉イベ
                        "クロック", "ラビット", "リーブス"]  # 事件簿
    additional_list2 = ["宝箱金", "宝箱銀", "宝箱銅"]  # 水着2021
    dropup_list = ["逆鱗", "心臓", "涙石",
                   "勲章", "貝殻", "蛇玉", "羽根", "蹄鉄", "頁", "歯車", "八連", "ランタン",
                   "種", "毒針", "塵", "牙", "火薬", "鉄杭", "髄液", "鎖", "骨", "証"]
    # 逆鱗 8501
    # 心臓 8500
    # 涙石 8402
    # 勲章 8309
    # 貝殻 8308
    # 蛇玉 8307
    # 羽根 8306
    # 蹄鉄 8305
    # 頁 8303
    # 歯車 8302
    # 八連 8301
    # ランタン 8300
    # 種 8203
    # 毒針  8202
    # 塵 8201
    # 牙 8200
    # 火薬 8105
    # 鉄杭 8104
    # 髄液 8103
    # 鎖 8102
    # 骨 8101
    # 証 8100
    print()
    for item in additional_list1:
        if item in output_items:
            print("追加出現率 %")
            break
    for item in additional_list2:
        if item in output_items:
            print("追加ドロップ率 %")
            break
    for item in dropup_list:
        if item in output_items:
            print(f"{item}泥UP %")
