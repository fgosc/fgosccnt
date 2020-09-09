#!/usr/bin/env python3
import os
import sys
import re
import argparse
from pathlib import Path
from collections import Counter
from enum import Enum
import itertools
import json
from operator import itemgetter
import math
import datetime
import logging
import multiprocessing
import time
import signal

import cv2
import numpy as np
import pytesseract
from PIL import Image
from PIL.ExifTags import TAGS

import pageinfo

PROGNAME = "FGOスクショカウント"
VERSION = "0.4.0"

logger = logging.getLogger(__name__)
watcher_running = True


class Ordering(Enum):
    """
        ファイルの処理順序を示す定数
    """
    NOTSPECIFIED = 'notspecified'   # 指定なし
    FILENAME = 'filename'           # ファイル名
    TIMESTAMP = 'timestamp'         # 作成日時

    def __str__(self):
        return str(self.value)


basedir = Path(__file__).resolve().parent
Item_dir = basedir / Path("item/equip/")
CE_dir = basedir / Path("item/ce/")
Point_dir = basedir / Path("item/point/")
train_item = basedir / Path("item.xml")  # アイテム下部
train_chest = basedir / Path("chest.xml")  # ドロップ数
train_card = basedir / Path("card.xml")  # ドロップ数
drop_file = basedir / Path("fgoscdata/hash_drop.json")
eventquest_dir = basedir / Path("fgoscdata/data/json/")

hasher = cv2.img_hash.PHash_create()

FONTSIZE_UNDEFINED = -1
FONTSIZE_NORMAL = 0
FONTSIZE_SMALL = 1
FONTSIZE_TINY = 2
PRIORITY_CE = 9000
PRIORITY_POINT = 3000
PRIORITY_ITEM = 700
PRIORITY_GEM_MIN = 6094
PRIORITY_MAGIC_GEM_MIN = 6194
PRIORITY_SECRET_GEM_MIN = 6294
PRIORITY_PIECE_MIN = 5194
PRIORITY_REWARD_QP = 9012
ID_START = 9500000
ID_QP = 1
ID_REWARD_QP = 5
ID_GEM_MIN = 6001
ID_GEM_MAX = 6007
ID_MAGIC_GEM_MIN = 6101
ID_MAGIC_GEM_MAX = 6107
ID_SECRET_GEM_MIN = 6201
ID_SECRET_GEM_MAX = 6207
ID_PIECE_MIN = 7001
ID_MONUMENT_MAX = 7107
ID_EXP_MIN = 9700100
ID_EXP_MAX = 9707500
ID_2ZORO_DICE = 94047708
ID_3ZORO_DICE = 94047709
ID_NORTH_AMERICA = 93000500
ID_SYURENJYO = 94006800
ID_EVNET = 94000000
TIMEOUT = 15
QP_UNKNOWN = -1
DEFAULT_POLL_FREQ = 60
DEFAULT_AMT_PROCESSES = 1


with open(drop_file, encoding='UTF-8') as f:
    drop_item = json.load(f)

# JSONファイルから各辞書を作成
item_name = {item["id"]: item["name"] for item in drop_item}
item_shortname = {item["id"]: item["shortname"] for item in drop_item
                  if "shortname" in item.keys()}
item_dropPriority = {item["id"]: item["dropPriority"] for item in drop_item}
item_type = {item["id"]: item["type"] for item in drop_item}
dist_item = {item["phash_battle"]: item["id"] for item in drop_item
             if item["type"] == "Item" and "phash_battle" in item.keys()}
dist_ce = {item["phash"]: item["id"] for item in drop_item
           if item["type"] == "Craft Essence"}
dist_secret_gem = {item["id"]: item["phash_class"] for item in drop_item
                   if 6200 < item["id"] < 6208
                   and "phash_class" in item.keys()}
dist_magic_gem = {item["id"]: item["phash_class"] for item in drop_item
                  if 6100 < item["id"] < 6108 and "phash_class" in item.keys()}
dist_gem = {item["id"]: item["phash_class"] for item in drop_item
            if 6000 < item["id"] < 6008 and "phash_class" in item.keys()}
dist_exp_rarity = {item["phash_rarity"]: item["id"] for item in drop_item
                   if item["type"] == "Exp. UP"
                   and "phash_rarity" in item.keys()}
dist_exp_rarity_sold = {item["phash_rarity_sold"]: item["id"] for item
                        in drop_item if item["type"] == "Exp. UP"
                        and "phash_rarity_sold" in item.keys()}
dist_exp_rarity.update(dist_exp_rarity_sold)
dist_exp_class = {item["phash_class"]: item["id"] for item in drop_item
                  if item["type"] == "Exp. UP"
                  and "phash_class" in item.keys()}
dist_exp_class_sold = {item["phash_class_sold"]: item["id"]
                       for item in drop_item
                       if item["type"] == "Exp. UP" and "phash_class_sold"
                       in item.keys()}
dist_exp_class.update(dist_exp_class_sold)
dist_point = {item["phash_battle"]: item["id"]
              for item in drop_item
              if item["type"] == "Point" and "phash_battle" in item.keys()}

with open(drop_file, encoding='UTF-8') as f:
    drop_item = json.load(f)

freequest = []
evnetfiles = eventquest_dir.glob('**/*.json')
for evnetfile in evnetfiles:
    try:
        with open(evnetfile, encoding='UTF-8') as f:
            event = json.load(f)
            freequest = freequest + event
    except:
        print("{}: ファイルが読み込めません".format(evnetfile))


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

    def __init__(self, img_rgb, svm, svm_chest, svm_card,
                 fileextention, debug=False, reward_only=False):
        TRAINING_IMG_WIDTH = 1755
        threshold = 80
        self.pagenum, self.pages, self.lines = pageinfo.guess_pageinfo(img_rgb)
        self.img_rgb_orig = img_rgb
        self.img_gray_orig = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        _, self.img_th_orig = cv2.threshold(self.img_gray_orig,
                                            threshold, 255, cv2.THRESH_BINARY)

        game_screen = self.extract_game_screen(debug)
        if debug:
            cv2.imwrite('game_screen.png', game_screen)

        _, width_g, _ = game_screen.shape
        wscale = (1.0 * width_g) / TRAINING_IMG_WIDTH
        resizeScale = 1 / wscale

        if resizeScale > 1:
            self.img_rgb = cv2.resize(game_screen, (0, 0),
                                      fx=resizeScale, fy=resizeScale,
                                      interpolation=cv2.INTER_CUBIC)
        else:
            self.img_rgb = cv2.resize(game_screen, (0, 0),
                                      fx=resizeScale, fy=resizeScale,
                                      interpolation=cv2.INTER_AREA)

        if debug:
            cv2.imwrite('game_screen_resize.png', self.img_rgb)

        mode = self.area_select()
        if debug:
            print("Area Mode: {}".format(mode))

        self.img_gray = cv2.cvtColor(self.img_rgb, cv2.COLOR_BGR2GRAY)
        _, self.img_th = cv2.threshold(self.img_gray,
                                       threshold, 255, cv2.THRESH_BINARY)
        self.svm = svm
        self.svm_chest = svm_chest

        self.height, self.width = self.img_rgb.shape[:2]
        self.chestnum = self.ocr_tresurechest(debug)
        if debug:
            print("総ドロップ数(OCR): {}".format(self.chestnum))
        item_pts = self.img2points()

        self.items = []
        self.current_dropPriority = PRIORITY_REWARD_QP
        if reward_only:
            # qpsplit.py で利用
            item_pts = item_pts[0:1]
        prev_item = None
        for i, pt in enumerate(item_pts):
            lx, _ = self.find_edge(self.img_th[pt[1]: pt[3],
                                               pt[0]: pt[2]], reverse=True)
            item_img_th = self.img_th[pt[1]: pt[3] - 30,
                                      pt[0] + lx: pt[2] + lx]
            if self.is_empty_box(item_img_th):
                break
            if debug:
                print("\n[Item{} Information]".format(i))
            item_img_rgb = self.img_rgb[pt[1]:  pt[3],
                                        pt[0] + lx:  pt[2] + lx]
            item_img_gray = self.img_gray[pt[1]: pt[3],
                                          pt[0] + lx: pt[2] + lx]
            if debug:
                cv2.imwrite('item' + str(i) + '.png', item_img_rgb)
            dropitem = Item(i, prev_item, item_img_rgb, item_img_gray,
                            svm, svm_card, fileextention,
                            self.current_dropPriority, mode, debug)
            if dropitem.id == -1:
                break
            self.current_dropPriority = item_dropPriority[dropitem.id]
            self.items.append(dropitem)
            prev_item = dropitem

        self.itemlist = self.makeitemlist()
        self.total_qp = self.get_qp(mode)
        self.qp_gained = self.get_qp_gained(mode, debug)
        self.scroll_position = self.determine_scroll_position(debug)

    def determine_scroll_position(self, debug=False):
        width = self.img_rgb.shape[1]
        # TODO: is it okay to hardcode this?
        topleft = (width - 90, 180)
        bottomright = (width, 180 + 660)

        if debug:
            img_copy = self.img_rgb.copy()
            cv2.rectangle(img_copy, topleft, bottomright, (0, 0, 255), 3)
            cv2.imwrite("./scroll_bar_selected.jpg", img_copy)

        gray_image = self.img_gray[topleft[1]: bottomright[1], topleft[0]: bottomright[0]]
        _, binary = cv2.threshold(gray_image, 225, 255, cv2.THRESH_BINARY)
        if debug:
            cv2.imwrite("scroll_bar_binary.png", binary)
        _, template = cv2.threshold(
            cv2.imread("./data/other/scroll_bar_upper.png",
                       cv2.IMREAD_GRAYSCALE),
            225,
            255,
            cv2.THRESH_BINARY,
        )

        res = cv2.matchTemplate(binary, template, cv2.TM_CCOEFF_NORMED)
        _, maxValue, _, max_loc = cv2.minMaxLoc(res)
        return max_loc[1] / gray_image.shape[0] if maxValue > 0.5 else -1

    def calc_black_whiteArea(self, bw_image):
        image_size = bw_image.size
        whitePixels = cv2.countNonZero(bw_image)

        whiteAreaRatio = (whitePixels / image_size) * 100  # [%]

        return whiteAreaRatio

    def is_empty_box(self, img_th):
        """
        アイテムボックスにアイテムが無いことを判別する
        """
        if self.calc_black_whiteArea(img_th) < 1:
            return True
        return False

    def get_qp_from_text(self, text):
        """
        capy-drop-parser から流用
        """
        qp = 0
        power = 1
        # re matches left to right so reverse the list
        # to process lower orders of magnitude first.
        for match in re.findall("[0-9]+", text)[::-1]:
            qp += int(match) * power
            power *= 1000

        return qp

    def extract_text_from_image(self, image):
        """
        capy-drop-parser から流用
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, qp_image = cv2.threshold(gray, 65, 255, cv2.THRESH_BINARY_INV)

        # '+' is needed to ensure that tesseract doesn't force a recognition on it,
        # which results in a '4' most of the time.
        return pytesseract.image_to_string(
            qp_image,
            config="-l eng --oem 1 --psm 7 -c tessedit_char_whitelist=+,0123456789",
        )

    def get_qp(self, mode):
        """
        capy-drop-parser から流用
        tesseract-OCR is quite slow and changed to use SVM
        """
        use_tesseract = False
        pt = pageinfo.detect_qp_region(self.img_rgb_orig, mode)
        logger.debug('pt from pageinfo: %s', pt)
        if pt is None:
            use_tesseract = True

        qp_total = -1
        if use_tesseract is False:  # use SVM
            im_th = cv2.bitwise_not(
                self.img_th_orig[pt[0][1]: pt[1][1], pt[0][0]: pt[1][0]]
            )
            qp_total = self.ocr_text(im_th)
        if use_tesseract or qp_total == -1:
            pt = ((288, 948), (838, 1024))
            logger.debug('Use tesseract')
            qp_total_text = self.extract_text_from_image(
                self.img_rgb[pt[0][1]: pt[1][1], pt[0][0]: pt[1][0]]
            )
            logger.debug('qp_total_text from text: %s', qp_total_text)
            qp_total = self.get_qp_from_text(qp_total_text)

        logger.debug('qp_total from text: %s', qp_total)
        if len(str(qp_total)) > 9:
            logger.warning(
                "qp_total exceeds the system's maximum: %s", qp_total
            )
        if qp_total == 0:
            return QP_UNKNOWN

        return qp_total

    def get_qp_gained(self, mode, debug=False):
        use_tesseract = False
        bounds = pageinfo.detect_qp_region(self.img_rgb_orig, mode)
        if bounds is None:
            # fall back on hardcoded bound
            bounds = ((398, 858), (948, 934))
            use_tesseract = True
        else:
            # Detecting the QP box with different shading is "easy", while detecting the absence of it
            # for the gain QP amount is hard. However, the 2 values have the same font and thus roughly
            # the same height (please NA...). You can consider them to be 2 same-sized boxes on top of
            # each other.
            (topleft, bottomright) = bounds
            height = bottomright[1] - topleft[1]
            topleft = (topleft[0], topleft[1] - height + int(height*0.12))
            bottomright = (bottomright[0], bottomright[1] - height)
            bounds = (topleft, bottomright)

        logger.debug('Gained QP bounds: %s', bounds)
        if debug:
            img_copy = self.img_rgb.copy()
            cv2.rectangle(img_copy, bounds[0], bounds[1], (0, 0, 255), 3)
            cv2.imwrite("./qp_gain_detection.jpg", img_copy)

        qp_gain = -1
        if use_tesseract is False:
            im_th = cv2.bitwise_not(
                self.img_th_orig[topleft[1]: bottomright[1],
                                 topleft[0]: bottomright[0]]
            )
            qp_gain = self.ocr_text(im_th)
        if use_tesseract or qp_gain == -1:
            logger.debug('Use tesseract')
            (topleft, bottomright) = bounds
            qp_gain_text = self.extract_text_from_image(
                self.img_rgb[topleft[1]: bottomright[1],
                             topleft[0]: bottomright[0]]
            )
            qp_gain = self.get_qp_from_text(qp_gain_text)
        logger.debug('qp from text: %s', qp_gain)
        if qp_gain == 0:
            qp_gain = QP_UNKNOWN

        return qp_gain

    def find_edge(self, img_th, reverse=False):
        """
        直線検出で検出されなかったフチ幅を検出
        """
        edge_width = 4
        _, width = img_th.shape[:2]
        target_color = 255 if reverse else 0
        for i in range(edge_width):
            img_th_x = img_th[:, i:i + 1]
            # ヒストグラムを計算
            hist = cv2.calcHist([img_th_x], [0], None, [256], [0, 256])
            # 最小値・最大値・最小値の位置・最大値の位置を取得
            _, _, _, maxLoc = cv2.minMaxLoc(hist)
            if maxLoc[1] == target_color:
                break
        lx = i
        for j in range(edge_width):
            img_th_x = img_th[:, width - j: width - j + 1]
            # ヒストグラムを計算
            hist = cv2.calcHist([img_th_x], [0], None, [256], [0, 256])
            # 最小値・最大値・最小値の位置・最大値の位置を取得
            _, _, _, maxLoc = cv2.minMaxLoc(hist)
            if maxLoc[1] == 0:
                break
        rx = i

        return lx, rx

    def extract_game_screen(self, debug=False):
        """
        1. Make cutting image using edge and line detection
        2. Correcting to be a gamescreen from cutting image
        """
        # 1. Edge detection
        height, width = self.img_gray_orig.shape[:2]
        canny_img = cv2.Canny(self.img_gray_orig, 100, 100)

        if debug:
            cv2.imwrite("canny_img.png", canny_img)

        # 2. Line detection
        # In the case where minLineLength is too short,
        # it catches the line of the item.
        lines = cv2.HoughLinesP(canny_img, rho=1, theta=np.pi/2,
                                threshold=80, minLineLength=int(height/5),
                                maxLineGap=6)

        left_x = upper_y = 0
        right_x = width
        bottom_y = height
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Detect Left line
            if x1 == x2 and x1 < width/2:
                if left_x < x1:
                    left_x = x1
            # Detect Upper line
            if y1 == y2 and y1 < height/2:
                if upper_y < y1:
                    upper_y = y1

        # Detect Right line
        # Avoid catching the line of the scroll bar
        if debug:
            line_img = self.img_rgb_orig.copy()

        for line in lines:
            x1, y1, x2, y2 = line[0]
            if debug:
                line_img = cv2.line(line_img, (x1, y1), (x2, y2),
                                    (0, 0, 255), 1)
                cv2.imwrite("line_img.png", line_img)
            if x1 == x2 and x1 > width*3/4 and (y1 < upper_y or y2 < upper_y):
                if right_x > x1:
                    right_x = x1

        # Detect Bottom line
        # Changed the underline of cut image to use the top of Next button.
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if y1 == y2 and y1 > height/2 and (x1 > right_x or x2 > right_x):
                if bottom_y > y1:
                    bottom_y = y1

        if debug:
            tmpimg = self.img_rgb_orig[upper_y: bottom_y, left_x: right_x]
            cv2.imwrite("cutting_img.png", tmpimg)
        # 内側の直線をとれなかったときのために補正する
        thimg = self.img_th_orig[upper_y: bottom_y, left_x: right_x]
        lx, rx = self.find_edge(thimg)
        left_x = left_x + lx
        right_x = right_x - rx

        # Correcting to be a gamescreen
        # Actual iPad (2048x1536) measurements
        scale = bottom_y - upper_y
        upper_y = upper_y - int(177*scale/847)
        bottom_y = bottom_y + int(124*scale/847)

        game_screen = self.img_rgb_orig[upper_y: bottom_y, left_x: right_x]
        return game_screen

    def area_select(self):
        """
        FGOアプリの地域を選択
        'na', 'jp'に対応

        'Next' '次へ'ボタンを読み込んで判別する
        """
        dist = {'jp': np.array([[198, 41, 185, 146, 50, 100, 140, 200]],
                               dtype='uint8'),
                'na': np.array([[70, 153, 57, 102, 6, 144, 148, 73]],
                               dtype='uint8')}
        img = self.img_rgb[1028:1134, 1416:1754]

        hash_img = hasher.compute(img)
        logger.debug("hash_img: %s", hash_img)
        hashorder = {}
        for i in dist.keys():
            dt = hasher.compare(hash_img, dist[i])
            hashorder[i] = dt
        hashorder = sorted(hashorder.items(), key=lambda x: x[1])
        logger.debug("hashorder: %s", hashorder)
        return next(iter(hashorder))[0]

    def makeitemlist(self):
        """
        アイテムを出力
        """
        itemlist = []
        for item in self.items:
            tmp = {}
            if item.category == "Quest Reward":
                tmp['id'] = ID_REWARD_QP
                tmp['name'] = "クエストクリア報酬QP"
                tmp['dropPriority'] = PRIORITY_REWARD_QP
            else:
                tmp['id'] = item.id
                tmp['name'] = item.name
                tmp['dropPriority'] = item_dropPriority[item.id]
            tmp['stack'] = int(item.dropnum[1:])
            tmp['bonus'] = item.bonus
            tmp['category'] = item.category
            tmp['x'] = item.position % 7
            tmp['y'] = item.position//7
            itemlist.append(tmp)
        return itemlist

    def ocr_text(self, im_th):
        h, w = im_th.shape[:2]
        # 物体検出
        im_th = cv2.bitwise_not(im_th)
        contours = cv2.findContours(im_th,
                                    cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)[0]
        item_pts = []
        for cnt in contours:
            ret = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            pt = [ret[0], ret[1], ret[0] + ret[2], ret[1] + ret[3]]
            if ret[2] < int(w/2) and area > 80 and ret[1] < h/2 \
                    and 0.4 < ret[2]/ret[3] < 0.85 and ret[3] > h/2:
                flag = False
                for p in item_pts:
                    if has_intersect(p, pt):
                        # どちらかを消す
                        p_area = (p[2]-p[0])*(p[3]-p[1])
                        pt_area = ret[2]*ret[3]
                        if p_area < pt_area:
                            item_pts.remove(p)
                        else:
                            flag = True

                if flag is False:
                    item_pts.append(pt)

        if len(item_pts) == 0:
            # Recognizing Failure
            return -1
        item_pts.sort()
        logger.debug("ドロップ桁数(OCR): %d", len(item_pts))

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
            hog = cv2.HOGDescriptor(win_size, block_size,
                                    block_stride, cell_size, bins)
            test.append(hog.compute(tmpimg))  # 特徴量の格納
            test = np.array(test)

            pred = self.svm_chest.predict(test)
            res = res + str(int(pred[1][0][0]))

        return int(res)

    def ocr_tresurechest(self, debug=False):
        """
        宝箱数をOCRする関数
        """
        pt = [1443, 20, 1505, 61]
        img_num = self.img_th[pt[1]:pt[3], pt[0]:pt[2]]
        im_th = cv2.bitwise_not(img_num)
        h, w = im_th.shape[:2]

        # 情報ウィンドウが数字とかぶった部分を除去する
        for y in range(h):
            im_th[y, 0] = 255
        for x in range(w):  # ドロップ数7のときバグる対策 #54
            im_th[0, x] = 255
        return self.ocr_text(im_th)

    def calc_offset(self, pts, std_pts, margin_x):
        """
        オフセットを反映
        """
        # Y列でソート
        pts.sort(key=lambda x: x[1])
        if len(pts) > 1:  # fix #107
            if (pts[1][3] - pts[1][1]) - (pts[0][3] - pts[0][1]) > 0:
                pts = pts[1:]
        # Offsetを算出
        offset_x = pts[0][0] - margin_x
        offset_y = pts[0][1] - std_pts[0][1]
        if offset_y > (std_pts[7][3] - std_pts[7][1])*2:
            # これ以上になったら三行目の座標と判断
            offset_y = pts[0][1] - std_pts[14][1]
        elif offset_y > 30:
            # これ以上になったら二行目の座標と判断
            offset_y = pts[0][1] - std_pts[7][1]

        # Offset を反映
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
        std_pts = self.booty_pts()

        row_size = 7  # アイテム表示最大列
        col_size = 3  # アイテム表示最大行
        margin_x = 15
        area_size_lower = 15000  # アイテム枠の面積の最小値
        img_1strow = self.img_th[0:self.height,
                                 std_pts[0][0] - margin_x:
                                 std_pts[0][2] + margin_x]

        # 輪郭を抽出
        contours = cv2.findContours(img_1strow, cv2.RETR_TREE,
                                    cv2.CHAIN_APPROX_SIMPLE)[0]

        leftcell_pts = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > area_size_lower \
               and area < self.height * self.width / (row_size * col_size):
                epsilon = 0.01*cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                if len(approx) == 6:  # 六角形のみ認識
                    ret = cv2.boundingRect(cnt)
                    if ret[1] > self.height * 0.15 \
                       and ret[1] + ret[3] < self.height * 0.76:
                        # 小数の数値はだいたいの実測
                        pts = [ret[0], ret[1],
                               ret[0] + ret[2], ret[1] + ret[3]]
                        leftcell_pts.append(pts)
        item_pts = self.calc_offset(leftcell_pts, std_pts, margin_x)

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
                                 item_width, item_height,
                                 margin_width, margin_height)
        return pts


def generate_booty_pts(criteria_left, criteria_top, item_width, item_height,
                       margin_width, margin_height):
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
    current = (criteria_left, criteria_top, criteria_left + item_width,
               criteria_top + item_height)
    for j in range(3):
        # top, bottom の y座標を計算
        current_top = criteria_top + (item_height + margin_height) * j
        current_bottom = current_top + item_height
        # x座標を左端に固定
        current = (criteria_left, current_top,
                   criteria_left + item_width, current_bottom)
        for i in range(7):
            # y座標を固定したままx座標をスライドさせていく
            current_left = criteria_left + (item_width + margin_width) * i
            current_right = current_left + item_width
            current = (current_left, current_top,
                       current_right, current_bottom)
            pts.append(current)
    return pts


class Item:
    def __init__(self, pos, prev_item, img_rgb, img_gray, svm, svm_card,
                 fileextention, current_dropPriority, mode='jp', debug=False):
        self.position = pos
        self.prev_item = prev_item
        self.img_rgb = img_rgb
        self.img_gray = img_gray
        self.img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)
        _, img_th = cv2.threshold(self.img_gray, 174, 255, cv2.THRESH_BINARY)
        self.img_th = cv2.bitwise_not(img_th)
        self.fileextention = fileextention

        self.height, self.width = img_rgb.shape[:2]
        self.identify_item(pos, prev_item, svm_card,
                           current_dropPriority, debug)
        if self.id == -1:
            return
        if debug:
            print("id: {}".format(self.id))
            print("dropPriority: {}".format(item_dropPriority[self.id]))
            print("Category: {}".format(self.category))
            print("Name: {}".format(self.name))

        self.svm = svm
        self.bonus = ""
        if self.category != "Craft Essence" and self.category != "Exp. UP":
            self.ocr_digit(mode, debug)
        else:
            self.dropnum = "x1"
        if debug:
            print("Number of Drop: {}".format(self.dropnum))

    def identify_item(self, pos, prev_item, svm_card,
                      current_dropPriority, debug):
        self.hash_item = compute_hash(self.img_rgb)  # 画像の距離
        if prev_item is not None:
            if not (prev_item.id == ID_REWARD_QP or
                    ID_GEM_MIN <= prev_item.id <= ID_SECRET_GEM_MAX or
                    ID_PIECE_MIN <= prev_item.id <= ID_MONUMENT_MAX or
                    ID_2ZORO_DICE <= prev_item.id <= ID_3ZORO_DICE or
                    ID_EXP_MIN <= prev_item.id <= ID_EXP_MAX):
                d = hasher.compare(self.hash_item, prev_item.hash_item)
                if d <= 4:
                    self.category = prev_item.category
                    self.id = prev_item.id
                    self.name = prev_item.name
                    return
        self.category = self.classify_category(svm_card)
        if pos < 14 and self.category == "":
            self.id = -1
            return
        self.id = self.classify_card(self.img_rgb, current_dropPriority, debug)
        self.name = item_name[self.id]
        if self.category == "":
            if self.id in item_type:
                self.category = item_type[self.id]
            else:
                self.category = "Item"

    def conflictcheck(self, pts, pt):
        """
        pt が ptsのどれかと衝突していたら面積に応じて入れ替える
        """
        flag = False
        for p in list(pts):
            if has_intersect(p, pt):
                # どちらかを消す
                p_area = (p[2]-p[0])*(p[3]-p[1])
                pt_area = (pt[2]-pt[0])*(pt[3]-pt[1])
                if p_area < pt_area:
                    pts.remove(p)
                else:
                    flag = True

        if flag is False:
            pts.append(pt)
        return pts

    def extension(self, pts):
        """
        文字エリアを1pixcel微修正
        """
        new_pts = []
        for pt in pts:
            if pt[0] == 0 and pt[1] == 0:
                pt = [pt[0], pt[1], pt[2], pt[3] + 1]
            elif pt[0] == 0 and pt[1] != 0:
                pt = [pt[0], pt[1] - 1, pt[2], pt[3] + 1]
            elif pt[0] != 0 and pt[1] == 0:
                pt = [pt[0] - 1, pt[1], pt[2], pt[3] + 1]
            else:
                pt = [pt[0] - 1, pt[1] - 1, pt[2], pt[3] + 1]
            new_pts.append(pt)
        return new_pts

    def extension_straighten(self, pts):
        """
        Y軸を最大値にそろえつつ文字エリアを1pixcel微修正
        """
        base_top = 6  # 強制的に高さを確保
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

    def detect_bonus_char4jpg(self, mode, debug):
        """
        戦利品数OCRで下段の黄文字の座標を抽出する
        PNGではない画像の認識用

        """
        # QP,ポイントはボーナス6桁のときに高さが変わる
        # それ以外は3桁のときに変わるはず(未確認)
        # ここのmargin_right はドロップ数の下一桁目までの距離
        base_line = 181 if mode == "na" else 179
        pattern_tiny = r"^\([\+x]\d{4,5}0\)$"
        pattern_tiny_qp = r"^\(\+\d{4,5}0\)$"
        pattern_small = r"^\([\+x]\d{5}0\)$"
        pattern_small_qp = r"^\(\+\d{5}0\)$"
        pattern_normal = r"^\([\+x]\d+\)$"
        pattern_normal_qp = r"^\(\+[1-9]\d+\)$"
        # 1-5桁の読み込み
        font_size = FONTSIZE_NORMAL
        if mode == 'na':
            margin_right = 20
        else:
            margin_right = 26
        line, pts = self.get_number4jpg(base_line, margin_right, font_size)
        if debug:
            print("BONUS NORMAL読み込み: {}".format(line))
        if self.name in ["QP", "ポイント"]:
            pattern_normal = pattern_normal_qp
        m_normal = re.match(pattern_normal, line)
        if m_normal:
            if debug:
                print("フォントサイズ: {}".format(font_size))
            return line, pts, font_size
        # 6桁の読み込み
        if mode == 'na':
            margin_right = 19
        else:
            margin_right = 25
        font_size = FONTSIZE_SMALL
        line, pts = self.get_number4jpg(base_line, margin_right, font_size)
        if debug:
            print("BONUS SMALL読み込み: {}".format(line))
        if self.name in ["QP", "ポイント"]:
            pattern_small = pattern_small_qp
        m_small = re.match(pattern_small, line)
        if m_small:
            if debug:
                print("フォントサイズ: {}".format(font_size))
            return line, pts, font_size
        # 7桁読み込み
        font_size = FONTSIZE_TINY
        if mode == 'na':
            margin_right = 18
        else:
            margin_right = 24
        line, pts = self.get_number4jpg(base_line, margin_right, font_size)
        if debug:
            print("BONUS TINY読み込み: {}".format(line))
        if self.name in ["QP", "ポイント"]:
            pattern_tiny = pattern_tiny_qp
        m_tiny = re.match(pattern_tiny, line)
        if m_tiny:
            if debug:
                print("Font Size: {}\nNumber of Drop:{}".format(font_size,
                                                                line))
            return line, pts, font_size
        else:
            font_size = FONTSIZE_UNDEFINED
            if debug:
                print("フォントサイズ: {}".format(font_size))
            line = ""
            pts = []

        return line, pts, font_size

    def detect_bonus_char(self):
        """
        戦利品数OCRで下段の黄文字の座標を抽出する

        HSVで黄色をマスクしてオブジェクト検出
        ノイズは少なく精度はかなり良い
        """

        margin_top = int(self.height*0.72)
        margin_bottom = int(self.height*0.11)
        margin_left = 8
        margin_right = 8

        img_hsv_lower = self.img_hsv[margin_top: self.height - margin_bottom,
                                     margin_left: self.width - margin_right]

        h, w = img_hsv_lower.shape[:2]
        # 手持ちスクショでうまくいっている範囲
        # 黄文字がこの数値でマスクできるかが肝
        # 未対応機種が発生したため[25,180,119] →[25,175,119]に変更
        lower_yellow = np.array([25, 175, 119])
        upper_yellow = np.array([37, 255, 255])

        img_hsv_lower_mask = cv2.inRange(img_hsv_lower,
                                         lower_yellow, upper_yellow)

        contours = cv2.findContours(img_hsv_lower_mask, cv2.RETR_TREE,
                                    cv2.CHAIN_APPROX_SIMPLE)[0]

        bonus_pts = []
        # 物体検出マスクがうまくいっているかが成功の全て
        for cnt in contours:
            ret = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            pt = [ret[0] + margin_left, ret[1] + margin_top,
                  ret[0] + ret[2] + margin_left, ret[1] + ret[3] + margin_top]

            # ）が上下に割れることがあるので上の一つは消す
            if ret[2] < int(w/2) and ret[1] < int(h*3/5) \
               and ret[1] + ret[3] > h*0.65 and area > 3:
                bonus_pts = self.conflictcheck(bonus_pts, pt)

        bonus_pts.sort()
        if len(bonus_pts) > 0:
            if self.width - bonus_pts[-1][2] > int((22*self.width/188)):
                # 黄文字は必ず右寄せなので最後の文字が画面端から離れている場合全部ゴミ
                bonus_pts = []

        return self.extension(bonus_pts)

    def define_fontsize(self, font_size):
        if font_size == FONTSIZE_NORMAL:
            cut_width = 20
            cut_height = 28
            comma_width = 9
        elif font_size == FONTSIZE_SMALL:
            cut_width = 18
            cut_height = 25
            comma_width = 8
        else:
            cut_width = 16
            cut_height = 22
            comma_width = 6
        return cut_width, cut_height, comma_width

    def get_number4jpg(self, base_line, margin_right, font_size):
        cut_width, cut_height, comma_width = self.define_fontsize(font_size)
        top_y = base_line - cut_height
        # まず、+, xの位置が何桁目か調査する
        pts = []
        if font_size == FONTSIZE_TINY:
            max_digits = 8
        elif font_size == FONTSIZE_SMALL:
            max_digits = 8
        else:
            max_digits = 7

        for i in range(max_digits):
            if i == 0:
                continue
            pt = [self.width - margin_right - cut_width * (i + 1)
                  - comma_width * int((i - 1)/3),
                  top_y,
                  self.width - margin_right - cut_width * i
                  - comma_width * int((i - 1)/3),
                  base_line]
            result = self.read_char(pt)
            if i == 1 and ord(result) == 0:
                # アイテム数 x1 とならず表記無し場合のエラー処理
                return "", pts
            if result in ['x', '+']:
                break
        # 決まった位置まで出力する
        line = ""
        for j in range(i):
            pt = [self.width - margin_right - cut_width * (j + 1)
                  - comma_width * int(j/3),
                  top_y,
                  self.width - margin_right - cut_width * j
                  - comma_width * int(j/3),
                  base_line]
            c = self.read_char(pt)
            if ord(c) == 0:  # Null文字対策
                line = line + '?'
                break
            line = line + c
            pts.append(pt)
        j = j + 1
        pt = [self.width - margin_right - cut_width * (j + 1)
              - comma_width * int((j - 1)/3),
              top_y,
              self.width - margin_right - cut_width * j
              - comma_width * int((j - 1)/3),
              base_line]
        c = self.read_char(pt)
        if ord(c) == 0:  # Null文字対策
            c = '?'
        line = line + c
        line = "(" + line[::-1] + ")"
        pts.append(pt)
        pts.sort()
        # PNGのマスク法との差を埋める補正
        new_pts = [[pts[0][0]-10, pts[0][1],
                    pts[0][0]-1, pts[0][3]]]  # "(" に対応
        new_pts.append("")  # ")" に対応

        return line, new_pts

    def get_number(self, base_line, margin_right, font_size):
        cut_width, cut_height, comma_width = self.define_fontsize(font_size)
        top_y = base_line - cut_height
        # まず、+, xの位置が何桁目か調査する
        for i in range(8):  # 8桁以上は無い
            if i == 0:
                continue
            elif (self.id == ID_REWARD_QP
                  or self.category in ["Point"]) and i <= 2:
                # 報酬QPとPointは3桁以上
                continue
            elif self.name == "QP" and i <= 3:
                # QPは4桁以上
                continue
            pt = [self.width - margin_right - cut_width * (i + 1)
                  - comma_width * int((i - 1)/3),
                  top_y,
                  self.width - margin_right - cut_width * i
                  - comma_width * int((i - 1)/3),
                  base_line]
            if pt[0] < 0:
                break
            result = self.read_char(pt)
            if i == 1 and ord(result) == 0:
                # アイテム数 x1 とならず表記無し場合のエラー処理
                return ""
            if result in ['x', '+']:
                break
        # 決まった位置まで出力する
        line = ""
        for j in range(i):
            if (self.id == ID_REWARD_QP) and j < 1:
                # 報酬QPの下一桁は0
                line += '0'
                continue
            elif (self.name == "QP" or self.category in ["Point"]) and j < 2:
                # QPとPointは下二桁は00
                line += '0'
                continue
            pt = [self.width - margin_right - cut_width * (j + 1)
                  - comma_width * int(j/3),
                  top_y,
                  self.width - margin_right - cut_width * j
                  - comma_width * int(j/3),
                  base_line]
            if pt[0] < 0:
                break
            c = self.read_char(pt)
            if ord(c) == 0:  # Null文字対策
                c = '?'
            line = line + c
        j = j + 1
        pt = [self.width - margin_right - cut_width * (j + 1)
              - comma_width * int((j - 1)/3),
              top_y,
              self.width - margin_right - cut_width * j
              - comma_width * int((j - 1)/3),
              base_line]
        if pt[0] > 0:
            c = self.read_char(pt)
            if ord(c) == 0:  # Null文字対策
                c = '?'
            line = line + c
        line = line[::-1]

        return line

    def detect_white_char(self, base_line, margin_right,
                          debug=False):
        """
        上段と下段の白文字を見つける機能を一つに統合
        """
        pattern_tiny = r"^[\+x][12]\d{4}00$"
        pattern_tiny_qp = r"^\+[12]\d{4}00$"
        pattern_small = r"^[\+x]\d{4}00$"
        pattern_small_qp = r"^\+\d{4}00$"
        pattern_normal = r"^[\+x][1-9]\d{0,5}$"
        pattern_normal_qp = r"^\+[1-9]\d{0,4}0$"
        if self.font_size != FONTSIZE_UNDEFINED:
            line = self.get_number(base_line, margin_right, self.font_size)
            if self.font_size == FONTSIZE_NORMAL:
                m_normal = re.match(pattern_normal, line)
                if m_normal:
                    return line
            elif self.font_size == FONTSIZE_SMALL:
                m_small = re.match(pattern_small, line)
                if m_small:
                    return line
            elif self.font_size == FONTSIZE_TINY:
                m_tiny = re.match(pattern_tiny, line)
                if m_tiny:
                    return line
            return ""
        else:
            # 1-6桁の読み込み
            font_size = FONTSIZE_NORMAL
            line = self.get_number(base_line, margin_right, font_size)
            if debug:
                print("NORMAL読み込み: {}".format(line))
            if self.name in ["QP", "ポイント"]:
                pattern_normal = pattern_normal_qp
            m_normal = re.match(pattern_normal, line)
            if m_normal:
                if debug:
                    print("Font Size: {}".format(font_size))
                self.font_size = font_size
                return line
            # 6桁の読み込み
            font_size = FONTSIZE_SMALL
            line = self.get_number(base_line, margin_right, font_size)
            if debug:
                print("SAMLL読み込み: {}".format(line))
            if self.name in ["QP", "ポイント"]:
                pattern_small = pattern_small_qp
            m_small = re.match(pattern_small, line)
            if m_small:
                if debug:
                    print("Font Size: {}".format(font_size))
                self.font_size = font_size
                return line
            # 7桁読み込み
            font_size = FONTSIZE_TINY
            line = self.get_number(base_line, margin_right, font_size)
            if debug:
                print("TINY読み込み: {}".format(line))
            if self.name in ["QP", "ポイント"]:
                pattern_tiny = pattern_tiny_qp
            m_tiny = re.match(pattern_tiny, line)
            if m_tiny:
                if debug:
                    print("Font Size: {}".format(font_size))
                self.font_size = font_size
                return line
            return ""

    def read_item(self, pts, debug=False):
        """
        ボーナスの数値をOCRする(エラー訂正有)
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
            hog = cv2.HOGDescriptor(win_size, block_size, block_stride,
                                    cell_size, bins)
            char.append(hog.compute(tmpimg))
            char = np.array(char)
            pred = self.svm.predict(char)
            result = int(pred[1][0][0])
            if result != 0:
                lines = lines + chr(result)
        if debug:
            print("OCR Result: {}".format(lines))
        # 以下エラー訂正
        if not lines.endswith(")"):
            lines = lines[:-1] + ")"
        if not lines.startswith("(+") and not lines.startswith("(x"):
            if lines[0] in ["+", 'x']:
                lines = "(" + lines
            else:
                lines = ""
        lines = lines.replace("()", "0")
        if len(lines) > 1:
            # エラー訂正 文字列左側
            # 主にイベントのポイントドロップで左側にゴミができるが、
            # 特定の記号がでてきたらそれより前はデータが無いはずなので削除する
            point_lbra = lines.rfind("(")
            point_plus = lines.rfind("+")
            point_x = lines.rfind("x")
            if point_lbra != -1:
                lines = lines[point_lbra:]
            elif point_plus != -1:
                lines = lines[point_plus:]
            elif point_x != -1:
                lines = lines[point_x:]

        if lines.isdigit():
            if int(lines) == 0:
                lines = "xErr"
            elif self.name == "QP" or self.name == "クエストクリア報酬QP":
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
        char = []
        tmpimg = self.img_gray[pt[1]:pt[3], pt[0]:pt[2]]
        tmpimg = cv2.resize(tmpimg, (win_size))
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride,
                                cell_size, bins)
        char.append(hog.compute(tmpimg))
        char = np.array(char)
        pred = self.svm.predict(char)
        result = int(pred[1][0][0])
        return chr(result)

    def ocr_digit(self, mode='jp', debug=False):
        """
        戦利品OCR
        """
        self.font_size = FONTSIZE_UNDEFINED

        if self.prev_item is None:
            prev_id = -1
        else:
            prev_id = self.prev_item.id

        if ID_GEM_MAX <= self.id <= ID_MONUMENT_MAX:
            # ボーナスが無いアイテム
            self.bonus_pts = []
            self.bonus = ""
            self.font_size = FONTSIZE_NORMAL
        elif prev_id == self.id \
                and self.category != "Point" and self.name != "QP":
            self.bonus_pts = self.prev_item.bonus_pts
            self.bonus = self.prev_item.bonus
            self.font_size = self.prev_item.font_size
        elif self.fileextention.lower() == '.png':
            self.bonus_pts = self.detect_bonus_char()
            self.bonus = self.read_item(self.bonus_pts, debug)
            # フォントサイズを決定
            if len(self.bonus_pts) > 0:
                y_height = self.bonus_pts[-1][3] - self.bonus_pts[-1][1]
                if y_height < 25:
                    self.font_size = FONTSIZE_TINY
                elif y_height > 27:
                    self.font_size = FONTSIZE_NORMAL
                else:
                    self.font_size = FONTSIZE_SMALL
        else:
            self.bonus, self.bonus_pts, self.font_size = self.detect_bonus_char4jpg(mode, debug)
        if debug:
            print("Bonus Font Size: {}\nBonus: {}".format(
                self.font_size, self.bonus))

        # 実際の(ボーナス無し)ドロップ数が上段にあるか下段にあるか決定
        offsset_y = 2 if mode == 'na' else 0
        if (self.category in ["Quest Reward", "Point"] or self.name == "QP") \
           and len(self.bonus) >= 5:  # ボーナスは"(+*0)"なので
            # 1桁目の上部からの距離を設定
            base_line = self.bonus_pts[-2][1] - 3 + offsset_y
        else:
            base_line = int(180/206*self.height)

        self.__bonus_string_into_int()

        # 実際の(ボーナス無し)ドロップ数の右端の位置を決定
        offset_x = -7 if mode == "na" else 0
        if self.category in ["Quest Reward", "Point"] or self.name == "QP":
            margin_right = 15 + offset_x
        elif len(self.bonus_pts) > 0:
            margin_right = self.width - self.bonus_pts[0][0] + 2
        else:
            margin_right = 15 + offset_x
        if debug:
            print("margin_right: {}".format(margin_right))
        self.dropnum = self.detect_white_char(base_line, margin_right,
                                              debug=debug)
        if len(self.dropnum) == 0:
            self.dropnum = "x1"

    def __bonus_string_into_int(self):
        try:
            self.bonus = int(re.sub(r"\(|\)|\+", "", self.bonus))
        except:
            self.bonus = 0

    def gem_img2id(self, img, gem_dict):
        hash_gem = self.compute_gem_hash(img)
        gems = {}
        for i in gem_dict.keys():
            d2 = hasher.compare(hash_gem, hex2hash(gem_dict[i]))
            if d2 <= 20:
                gems[i] = d2
        gems = sorted(gems.items(), key=lambda x: x[1])
        gem = next(iter(gems))
        return gem[0]

    def classify_item(self, img, currnet_dropPriority, debug=False):
        """
        imgとの距離を比較して近いアイテムを求める
        id を返すように変更
        """
        hash_item = compute_hash(img)  # 画像の距離
        ids = {}
        if debug:
            hex = ""
            for h in hash_item[0]:
                hex = hex + "{:02x}".format(h)
            print("phash :{}".format(hex))
        # 既存のアイテムとの距離を比較
        for i in dist_item.keys():
            d = hasher.compare(hash_item, hex2hash(i))
            if d <= 12:
                # ポイントと種の距離が8という例有り(IMG_0274)→16に
                # バーガーと脂の距離が10という例有り(IMG_2354)→14に
                ids[dist_item[i]] = d
        if len(ids) > 0:
            ids = sorted(ids.items(), key=lambda x: x[1])
            id_tupple = next(iter(ids))
            id = id_tupple[0]
            if ID_SECRET_GEM_MIN <= id <= ID_SECRET_GEM_MAX:
                if currnet_dropPriority >= PRIORITY_SECRET_GEM_MIN:
                    id = self.gem_img2id(img, dist_secret_gem)
                else:
                    return ""
            elif ID_MAGIC_GEM_MIN <= id <= ID_MAGIC_GEM_MAX:
                if currnet_dropPriority >= PRIORITY_MAGIC_GEM_MIN:
                    id = self.gem_img2id(img, dist_magic_gem)
                else:
                    return ""
            elif ID_GEM_MIN <= id <= ID_GEM_MAX:
                if currnet_dropPriority >= PRIORITY_GEM_MIN:
                    id = self.gem_img2id(img, dist_gem)
                else:
                    return ""
            elif ID_PIECE_MIN <= id <= ID_MONUMENT_MAX:
                if currnet_dropPriority < PRIORITY_PIECE_MIN:
                    return ""
                # ヒストグラム
                img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                h, w = img_hsv.shape[:2]
                img_hsv = img_hsv[int(h/2-10): int(h/2+10),
                                  int(w/2-10): int(w/2+10)]
                hist_s = cv2.calcHist([img_hsv], [1], None,
                                      [256], [0, 256])  # Bのヒストグラムを計算
                _, _, _, maxLoc = cv2.minMaxLoc(hist_s)
                if maxLoc[1] > 128:
                    id = int(str(id)[0] + "1" + str(id)[2:])
                else:
                    id = int(str(id)[0] + "0" + str(id)[2:])

            return id

        return ""

    def classify_ce(self, img, debug=False):
        """
        imgとの距離を比較して近いアイテムを求める
        """
        hash_item = compute_hash_ce(img)  # 画像の距離
        itemfiles = {}
        if debug:
            hex = ""
            for h in hash_item[0]:
                hex = hex + "{:02x}".format(h)
            print("phash :{}".format(hex))
        # 既存のアイテムとの距離を比較
        for i in dist_ce.keys():
            d = hasher.compare(hash_item, hex2hash(i))
            if d <= 12:
                itemfiles[dist_ce[i]] = d
        if len(itemfiles) > 0:
            itemfiles = sorted(itemfiles.items(), key=lambda x: x[1])
            item = next(iter(itemfiles))

            return item[0]

        return ""

    def classify_point(self, img, debug=False):
        """
        imgとの距離を比較して近いアイテムを求める
        """
        hash_item = compute_hash(img)  # 画像の距離
        itemfiles = {}
        if debug:
            hex = ""
            for h in hash_item[0]:
                hex = hex + "{:02x}".format(h)
            print("phash :{}".format(hex))
        # 既存のアイテムとの距離を比較
        for i in dist_point.keys():
            d = hasher.compare(hash_item, hex2hash(i))
            if d <= 12:
                itemfiles[dist_point[i]] = d
        if len(itemfiles) > 0:
            itemfiles = sorted(itemfiles.items(), key=lambda x: x[1])
            item = next(iter(itemfiles))

            return item[0]

        return ""

    def classify_exp(self, img):
        hash_item = self.compute_exp_rarity_hash(img)  # 画像の距離
        exps = {}
        for i in dist_exp_rarity.keys():
            dt = hasher.compare(hash_item, hex2hash(i))
            if dt <= 15:  # IMG_1833で11 IMG_1837で15
                exps[i] = dt
        exps = sorted(exps.items(), key=lambda x: x[1])
        if len(exps) > 0:
            exp = next(iter(exps))

            hash_exp_class = self.compute_exp_class_hash(img)
            exp_classes = {}
            for j in dist_exp_class.keys():
                dtc = hasher.compare(hash_exp_class, hex2hash(j))
                exp_classes[j] = dtc
            exp_classes = sorted(exp_classes.items(), key=lambda x: x[1])
            exp_class = next(iter(exp_classes))

            return int(str(dist_exp_class[exp_class[0]])[:4]
                       + str(dist_exp_rarity[exp[0]])[4] + "00")

        return ""

    def make_new_file(self, img, search_dir, dist_dic, dropPriority, category):
        """
        ファイル名候補を探す
        """
        i_dic = {"Item": "item", "Craft Essence": "ce", "Point": "point"}
        initial = i_dic[category]
        for i in range(999):
            itemfile = search_dir / (initial + '{:0=3}'.format(i + 1) + '.png')
            if itemfile.is_file():
                continue
            else:
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(itemfile.as_posix(), img_gray)
                # id 候補を決める
                for j in range(99999):
                    id = j + ID_START
                    if id in item_name.keys():
                        continue
                    break
                if category == "Craft Essence":
                    hash = compute_hash_ce(img)
                else:
                    hash = compute_hash(img)
                hash_hex = ""
                for h in hash[0]:
                    hash_hex = hash_hex + "{:02x}".format(h)
                dist_dic[hash_hex] = id
                item_name[id] = itemfile.stem
                item_dropPriority[id] = dropPriority
                item_type[id] = category
                break
        return id

    def classify_category(self, svm_card):
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
        carddic = {0: 'Quest Reward', 1: 'Item', 2: 'Point',
                   3: 'Craft Essence', 4: 'Exp. UP', 99: ""}

        tmpimg = self.img_rgb[int(189/206*self.height):
                              int(201/206*self.height),
                              int(78/188*self.width):
                              int(115/188*self.width)]

        tmpimg = cv2.resize(tmpimg, (win_size))
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride,
                                cell_size, bins)
        test.append(hog.compute(tmpimg))  # 特徴量の格納
        test = np.array(test)
        pred = svm_card.predict(test)

        return carddic[pred[1][0][0]]

    def classify_card(self, img, currnet_dropPriority, debug=False):
        """
        アイテム判別器
        """
        if self.category == "Point":
            id = self.classify_point(img, debug)
            if id == "":
                id = self.make_new_file(img, Point_dir, dist_point,
                                        PRIORITY_POINT, self.category)
            return id
        elif self.category == "Quest Reward":
            return 5
        elif self.category == "Craft Essence":
            id = self.classify_ce(img, debug)
            if id == "":
                id = self.make_new_file(img, CE_dir, dist_ce,
                                        PRIORITY_CE, self.category)
            return id
        elif self.category == "Exp. UP":
            return self.classify_exp(img)
        elif self.category == "Item":
            id = self.classify_item(img, currnet_dropPriority, debug)
            if id == "":
                id = self.make_new_file(img, Item_dir, dist_item,
                                        PRIORITY_ITEM, self.category)
        else:
            # ここで category が判別できないのは三行目かつ
            # スクロール位置の関係で下部表示が消えている場合
            id = self.classify_item(img, currnet_dropPriority, debug)
            if id != "":
                return id
            id = self.classify_point(img, debug)
            if id != "":
                return id
            id = self.classify_ce(img, debug)
            if id != "":
                return id
            id = self.classify_exp(img)
            if id != "":
                return id
        if id == "":
            id = self.make_new_file(img, Item_dir, dist_item,
                                    PRIORITY_ITEM, "Item")
        return id

    def compute_exp_rarity_hash(self, img_rgb):
        """
        種火レアリティ判別器
        この場合は画像全域のハッシュをとる
        """
        img = img_rgb[int(53/189*self.height):int(136/189*self.height),
                      int(37/206*self.width):int(149/206*self.width)]

        return hasher.compute(img)

    def compute_exp_class_hash(self, img_rgb):
        """
        種火クラス判別器
        左上のクラスマークぎりぎりのハッシュを取る
        記述した比率はiPhone6S画像の実測値
        """
        img = img_rgb[int(5/135*self.height):int(30/135*self.height),
                      int(5/135*self.width):int(30/135*self.width)]
        return hasher.compute(img)

    def compute_gem_hash(self, img_rgb):
        """
        スキル石クラス判別器
        中央のクラスマークぎりぎりのハッシュを取る
        記述した比率はiPhone6S画像の実測値
        """
        height, width = img_rgb.shape[:2]

        img = img_rgb[int((145-16-60*0.8)/2/145*height)+3:
                      int((145-16+60*0.8)/2/145*height)+3,
                      int((132-52*0.8)/2/132*width):
                      int((132+52*0.8)/2/132*width)]

        return hasher.compute(img)


def compute_hash(img_rgb):
    """
    判別器
    この判別器は下部のドロップ数を除いた部分を比較するもの
    記述した比率はiPhone6S画像の実測値
    """
    height, width = img_rgb.shape[:2]
    img = img_rgb[int(22/135*height):
                  int(77/135*height),
                  int(23/135*width):
                  int(112/135*width)]
    return hasher.compute(img)


def compute_hash_ce(img_rgb):
    """
    判別器
    この判別器は下部のドロップ数を除いた部分を比較するもの
    記述した比率はiPpd2018画像の実測値
    """
    img = img_rgb[12:176, 9:182]
    return hasher.compute(img)


def search_file(search_dir, dist_dic, dropPriority, category):
    """
    Item, Craft Essence, Pointの各ファイルを探す
    """
    files = search_dir.glob('**/*.png')
    for fname in files:
        img = imread(fname)
        # id 候補を決める
        # 既存のデータがあったらそれを使用
        if fname.stem in item_name.values():
            id = [k for k, v in item_name.items() if v == fname.stem][0]
        elif fname.stem in item_shortname.values():
            id = [k for k, v in item_shortname.items() if v == fname.stem][0]
        else:
            for j in range(99999):
                id = j + ID_START
                if id in item_name.keys():
                    continue
                break
        # priotiry は固定
            item_name[id] = fname.stem
            item_dropPriority[id] = dropPriority
            item_type[id] = category
        if category == "Craft Essence":
            hash = compute_hash_ce(img)
        else:
            hash = compute_hash(img)
        hash_hex = ""
        for h in hash[0]:
            hash_hex = hash_hex + "{:02x}".format(h)
        dist_dic[hash_hex] = id


def calc_dist_local():
    """
    既所持のアイテム画像の距離(一次元配列)の辞書を作成して保持
    """
    search_file(Item_dir, dist_item, PRIORITY_ITEM, "Item")
    search_file(CE_dir, dist_ce, PRIORITY_CE, "Craft Essence")
    search_file(Point_dir, dist_point, PRIORITY_POINT, "Point")


def hex2hash(hexstr):
    hashlist = []
    for i in range(8):
        hashlist.append(int('0x' + hexstr[i*2:i*2+2], 0))
    return np.array([hashlist], dtype='uint8')


def out_name(id):
    if id in item_shortname.keys():
        name = item_shortname[id]
    else:
        name = item_name[id]
    return name


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


def get_exif(img):
    exif = img._getexif()
    try:
        for id, val in exif.items():
            tg = TAGS.get(id, id)
            if tg == "DateTimeOriginal":
                return datetime.datetime.strptime(val, '%Y:%m:%d %H:%M:%S')
    except AttributeError:
        return "NON"
    return "NON"


def get_output(filenames, args):
    """
    出力内容を作成
    """
    debug = args.debug
    calc_dist_local()
    if train_item.exists() is False:
        print("[エラー]item.xml が存在しません")
        print("python makeitem.py を実行してください")
        sys.exit(1)
    if train_chest.exists() is False:
        print("[エラー]chest.xml が存在しません")
        print("python makechest.py を実行してください")
        sys.exit(1)
    if train_card.exists() is False:
        print("[エラー]card.xml が存在しません")
        print("python makecard.py を実行してください")
        sys.exit(1)
    svm = cv2.ml.SVM_load(str(train_item))
    svm_chest = cv2.ml.SVM_load(str(train_chest))
    svm_card = cv2.ml.SVM_load(str(train_card))

    fileoutput = []  # 出力
    prev_pages = 0
    prev_pagenum = 0
    prev_total_qp = QP_UNKNOWN
    prev_itemlist = []
    prev_datetime = datetime.datetime(year=2015, month=7, day=30, hour=0)
    all_list = []

    for filename in filenames:
        if debug:
            print(filename)
        f = Path(filename)

        if f.exists() is False:
            output = {'filename': str(filename) + ': not found'}
            all_list.append([])
        else:
            img_rgb = imread(filename)
            fileextention = Path(filename).suffix

            try:
                sc = ScreenShot(img_rgb,
                                svm, svm_chest, svm_card,
                                fileextention, debug)

                # ドロップ内容が同じで下記のとき、重複除外
                # QPカンストじゃない時、QPが前と一緒
                # QPカンストの時、Exif内のファイル作成時間が15秒未満
                pilimg = Image.open(filename)
                dt = get_exif(pilimg)
                if dt == "NON" or prev_datetime == "NON":
                    td = datetime.timedelta(days=1)
                else:
                    td = dt - prev_datetime
                if sc.pages - sc.pagenum == 0:
                    sc.itemlist = sc.itemlist[14-(sc.lines+2) % 3*7:]
                if prev_itemlist == sc.itemlist:
                    if (sc.total_qp != 999999999
                        and sc.total_qp == prev_total_qp) \
                        or (sc.total_qp == 999999999
                            and td.total_seconds() < args.timeout):
                        if debug:
                            print("args.timeout: {}".format(args.timeout))
                            print("filename: {}".format(filename))
                            print("prev_itemlist: {}".format(prev_itemlist))
                            print("sc.itemlist: {}".format(sc.itemlist))
                            print("sc.total_qp: {}".format(sc.total_qp))
                            print("prev_total_qp: {}".format(prev_total_qp))
                            print("datetime: {}".format(dt))
                            print("prev_datetime: {}".format(prev_datetime))
                            print("td.total_second: {}".format(
                                td.total_seconds()))
                        fileoutput.append(
                            {'filename': str(filename) + ': duplicate'})
                        all_list.append([])
                        continue

                # 2頁目以前のスクショが無い場合に migging と出力
                if (prev_pages - prev_pagenum > 0
                    and sc.pagenum - prev_pagenum != 1) \
                   or (prev_pages - prev_pagenum == 0 and sc.pagenum != 1):
                    fileoutput.append({'filename': 'missing'})
                    all_list.append([])

                all_list.append(sc.itemlist)

                prev_pages = sc.pages
                prev_pagenum = sc.pagenum
                prev_total_qp = sc.total_qp
                prev_itemlist = sc.itemlist
                prev_datetime = dt

                sumdrop = len([d for d in sc.itemlist
                               if d["name"] != "クエストクリア報酬QP"])
                output = {'filename': str(filename), 'ドロ数': sumdrop}
                if sc.pagenum == 1:
                    if sc.lines >= 7:
                        output["ドロ数"] = str(output["ドロ数"]) + "++"
                    elif sc.lines >= 4:
                        output["ドロ数"] = str(output["ドロ数"]) + "+"
                elif sc.pagenum == 2 and sc.lines >= 7:
                    output["ドロ数"] = str(output["ドロ数"]) + "+"

            except Exception as e:
                logger.error(e, exc_info=True)
                output = ({'filename': str(filename) + ': not valid'})
                all_list.append([])
        fileoutput.append(output)
    return fileoutput, all_list


def load_svms():
    svm = cv2.ml.SVM_load(str(train_item))
    svm_chest = cv2.ml.SVM_load(str(train_chest))
    svm_card = cv2.ml.SVM_load(str(train_card))
    return (svm, svm_chest, svm_card)


def parse_img(
        svm,
        svm_chest,
        svm_card,
        file_path,
        prev_pages=0,
        prev_pagenum=0,
        prev_total_qp=QP_UNKNOWN,
        prev_gained_qp=QP_UNKNOWN,
        prev_itemlist=[],
        prev_datetime=datetime.datetime(year=2015, month=7, day=30, hour=0),
        debug=False):
    parsed_img_data = {"status": "Incomplete"}

    if debug:
        print(file_path)
    parsed_img_data["image_path"] = str(os.path.abspath(file_path))

    if not Path(file_path).exists():
        # TODO: is this needed?
        parsed_img_data["status"] = "File not found"
        return parsed_img_data

    img_rgb = imread(file_path)
    file_extention = Path(file_path).suffix

    try:
        screenshot = ScreenShot(
            img_rgb, svm, svm_chest, svm_card, file_extention, debug)

        # If the previous image indicated more coming, check whether this is the fated one.
        if (prev_pages - prev_pagenum > 0 and screenshot.pagenum - prev_pagenum != 1) \
                or (prev_pages - prev_pagenum == 0 and screenshot.pagenum != 1):
            parsed_img_data["status"] = "Missing page before this"

        # Detect whether image is a duplicate
        # Image is a candidate duplicate if drops and gained QP match previous image.
        # Duplicate is confirmed if:
        # - QP is not capped and drops are the same as in the previous image
        # - QP is capped and previous image was taken within 15sec
        # TODO: is this needed?
        pilimg = Image.open(file_path)
        date_time = get_exif(pilimg)
        if date_time == "NON" or prev_datetime == "NON":
            time_delta = datetime.timedelta(days=1)
        else:
            time_delta = date_time - prev_datetime
        if prev_itemlist == screenshot.itemlist and prev_gained_qp == screenshot.qp_gained:
            if (screenshot.total_qp != 999999999 and screenshot.total_qp == prev_total_qp) \
                    or (screenshot.total_qp == 999999999 and time_delta.total_seconds() < args.timeout):
                if debug:
                    print("args.timeout: {}".format(args.timeout))
                    print("filename: {}".format(file_path))
                    print("prev_itemlist: {}".format(prev_itemlist))
                    print("screenshot.itemlist: {}".format(
                        screenshot.itemlist))
                    print("screenshot.total_qp: {}".format(
                        screenshot.total_qp))
                    print("prev_total_qp: {}".format(prev_total_qp))
                    print("datetime: {}".format(date_time))
                    print("prev_datetime: {}".format(prev_datetime))
                    print("td.total_second: {}".format(
                        time_delta.total_seconds()))
                parsed_img_data["status"] = "Duplicate file"
                return parsed_img_data

        # Prep next iter
        prev_pages = screenshot.pages
        prev_pagenum = screenshot.pagenum
        prev_total_qp = screenshot.total_qp
        prev_gained_qp = screenshot.qp_gained
        prev_itemlist = screenshot.itemlist
        prev_datetime = date_time

        # Gather data
        parsed_img_data["qp_total"] = screenshot.total_qp
        parsed_img_data["qp_gained"] = screenshot.qp_gained
        parsed_img_data["scroll_position"] = screenshot.scroll_position
        parsed_img_data["drop_count"] = screenshot.chestnum
        parsed_img_data["drops_found"] = len(screenshot.itemlist)
        parsed_img_data["drops"] = screenshot.itemlist
        parsed_img_data["status"] = "OK" if parsed_img_data["status"] == "Incomplete" else parsed_img_data["status"]
        return parsed_img_data

    except Exception as e:
        logger.error("Error during parsing of {}\n{}\n".format(
            file_path, e), exc_info=True)
        parsed_img_data["status"] = "Invalid file"
        return parsed_img_data


def move_file_to_out_dir(src_file_path, out_dir):
    if out_dir is not None:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        src_file_path = Path(src_file_path)
        if not src_file_path.exists():
            print("Cannot move {}. It does not exist.".format(src_file_path))
            exit(1)

        dst_file_path = "{}/{}".format(out_dir, src_file_path.name)
        os.rename(src_file_path, dst_file_path)

        return dst_file_path

    return src_file_path


def check_svms_trained():
    if train_item.exists() is False:
        print("[エラー]item.xml が存在しません")
        print("python makeitem.py を実行してください")
        sys.exit(1)
    if train_chest.exists() is False:
        print("[エラー]chest.xml が存在しません")
        print("python makechest.py を実行してください")
        sys.exit(1)
    if train_card.exists() is False:
        print("[エラー]card.xml が存在しません")
        print("python makecard.py を実行してください")
        sys.exit(1)


def parse_into_json(input_file_paths, args):
    """
    The version of output gathering used by AtlasAcademy. Made to resemble capy's output.
    """
    debug = args.debug

    calc_dist_local()
    check_svms_trained()

    (svm, svm_chest, svm_card) = load_svms()

    prev_pages = 0
    prev_pagenum = 0
    prev_total_qp = QP_UNKNOWN
    prev_gained_qp = QP_UNKNOWN
    prev_itemlist = []
    prev_datetime = datetime.datetime(year=2015, month=7, day=30, hour=0)
    all_parsed_output = []

    for file_path in input_file_paths:
        file_path = move_file_to_out_dir(file_path, args.out_folder)
        all_parsed_output.append(parse_img(
            svm,
            svm_chest,
            svm_card,
            file_path,
            prev_pages,
            prev_pagenum,
            prev_total_qp,
            prev_gained_qp,
            prev_itemlist,
            prev_datetime,
            debug))
    return all_parsed_output


def __parse_into_json_process(input_queue, args):
    (svm, svm_chest, svm_card) = load_svms()

    global watcher_running
    while watcher_running or not input_queue.empty():
        input_file_path = input_queue.get()
        # Detection of missing screenshots/pages (e.g. scrolled down image with no previous
        # image to go along with it), is dissabled with `prev_pages=-1`. This is because
        # the technique depends on having the images sorted in chronological order. Sorting
        # files and processing them in order is not possible in a multiprocess environment.
        parsed_output = parse_img(
            svm,
            svm_chest,
            svm_card,
            input_file_path,
            prev_pages=-1,
            debug=args.debug)
        output_json([parsed_output], args.out_folder)


def __signal_handling(*_):
    """
    Taken from capy-drop-parser
    """
    global watcher_running
    if not watcher_running:
        sys.exit(1)
    watcher_running = False
    print(
        "Notice: app may take up to polling frequency time and however long it takes to finish the queue before exiting."
    )


def watch_parse_output_into_json(args):
    """
    Continuously watch the given input directory for new files.
    Processes any new images by parsing them, moving them to output dir, and writing parsed json to
    output dir.

    Works with a producer/consumer multiprocessing approach. This function watches and
    fills the queue, while spawned processes use `__parse_into_json_process` to consume the
    items.
    """
    calc_dist_local()
    check_svms_trained()
    signal.signal(signal.SIGINT, __signal_handling)

    # We estimate roughly 2secs per image parsing. Queue can hold as many images as can be
    # processed by the given amount of processes in the given amount of poll time.
    input_queue = multiprocessing.Queue(maxsize=int(
        args.num_processes * args.polling_frequency / 2))
    pool = multiprocessing.Pool(
        args.num_processes, initializer=__parse_into_json_process, initargs=(input_queue, args))

    global watcher_running
    while watcher_running:
        for f in Path(args.folder).iterdir():
            if not f.is_file():
                continue

            file_path = move_file_to_out_dir(f, args.out_folder)
            input_queue.put(file_path)  # blocks when queue is full

        time.sleep(int(args.polling_frequency))

    input_queue.close()
    input_queue.join_thread()
    pool.close()
    pool.join()


def sort_files(files, ordering):
    if ordering == Ordering.NOTSPECIFIED:
        return files
    elif ordering == Ordering.FILENAME:
        return sorted(files)
    elif ordering == Ordering.TIMESTAMP:
        return sorted(files, key=lambda f: Path(f).stat().st_ctime)
    raise ValueError(f'Unsupported ordering: {ordering}')


def change_value(line):
    line = re.sub('000000$', "百万", str(line))
    line = re.sub('0000$', "万", str(line))
    line = re.sub('000$', "千", str(line))
    return line


def make_quest_output(quest):
    output = ""
    if quest != "":
        quest_list = [q["name"] for q in freequest
                      if q["place"] == quest["place"]]
        if math.floor(quest["id"]/100)*100 == ID_NORTH_AMERICA:
            output = quest["place"] + " " + quest["name"]
        elif math.floor(quest["id"]/100)*100 == ID_SYURENJYO:
            output = quest["chapter"] + " " + quest["place"]
        elif math.floor(quest["id"]/100000)*100000 == ID_EVNET:
            output = quest["shortname"]
        else:
            # クエストが0番目のときは場所を出力、それ以外はクエスト名を出力
            if quest_list.index(quest["name"]) == 0:
                output = quest["chapter"] + " " + quest["place"]
            else:
                output = quest["chapter"] + " " + quest["name"]
    return output


def deside_quest(item_list):

    item_set = set()
    for item in item_list:
        if item["id"] == 5:
            item_set.add("QP(+" + str(item["dropnum"]) + ")")
        elif item["id"] == 1 \
            or item["category"] == "Craft Essence" \
            or (9700 <= math.floor(item["id"]/1000) <= 9707
                and str(item["id"])[4] not in ["4", "5"]):
            continue
        else:
            item_set.add(item["name"])
    quest_candidate = ""
    for quest in reversed(freequest):
        dropset = {i["name"] for i in quest["drop"]
                   if i["type"] != "Craft Essence"}
        dropset.add("QP(+" + str(quest["qp"]) + ")")
        if dropset == item_set:
            quest_candidate = quest
            break
    return quest_candidate


def make_csv_header(item_list):
    """
    CSVのヘッダ情報を作成
    礼装のドロップが無いかつ恒常以外のアイテムが有るとき礼装0をつける
    """
    if item_list == [[]]:
        return ['filename', 'ドロ数'], False, ""
    # リストを一次元に
    flat_list = list(itertools.chain.from_iterable(item_list))
    # 余計な要素を除く
    short_list = [{"id": a["id"], "name": a["name"], "category": a["category"],
                   "dropPriority": a["dropPriority"], "dropnum": a["dropnum"]}
                  for a in flat_list]
    ce0_flag = ("Craft Essence"
                not in [d.get('category') for d in flat_list]) \
        and (max([d.get("id") for d in flat_list]) > 9707500)
    if ce0_flag:
        short_list.append({"id": 99999990, "name": "礼装",
                           "category": "Craft Essence",
                           "dropPriority": 9000, "dropnum": 0})
    # 重複する要素を除く
    unique_list = list(map(json.loads, set(map(json.dumps, short_list))))
    # ソート
    new_list = sorted(sorted(sorted(unique_list, key=itemgetter('dropnum')),
                             key=itemgetter('id'), reverse=True),
                      key=itemgetter('dropPriority'), reverse=True)
    header = []
    for nlist in new_list:
        if nlist['category'] in ['Quest Reward', 'Point'] \
           or nlist["name"] == "QP":
            tmp = out_name(nlist['id']) \
                + "(+" + change_value(nlist["dropnum"]) + ")"
        elif nlist["dropnum"] > 1:
            tmp = out_name(nlist['id']) \
                + "(x" + change_value(nlist["dropnum"]) + ")"
        elif nlist["name"] == "礼装":
            tmp = "礼装"
        else:
            tmp = out_name(nlist['id'])
        header.append(tmp)
    # クエスト名判定
    quest = deside_quest(new_list)
    quest_output = make_quest_output(quest)
    return ['filename', 'ドロ数'] + header, ce0_flag, quest_output


def make_csv_data(sc_list, ce0_flag):
    if sc_list == []:
        return [{}], [{}]
    csv_data = []
    allitem = []
    for sc in sc_list:
        tmp = []
        for item in sc:
            if item['category'] in ['Quest Reward', 'Point'] \
               or item["name"] == "QP":
                tmp.append(out_name(item['id'])
                           + "(+" + change_value(item["dropnum"]) + ")")
            elif item["dropnum"] > 1:
                tmp.append(out_name(item['id'])
                           + "(x" + change_value(item["dropnum"]) + ")")
            else:
                tmp.append(out_name(item['id']))
        allitem = allitem + tmp
        csv_data.append(dict(Counter(tmp)))
    csv_sum = dict(Counter(allitem))
    if ce0_flag:
        csv_sum.update({"礼装": 0})
    return csv_sum, csv_data


def output_json(parsed_output, out_folder):
    if out_folder is None:
        sys.stdout.buffer.write(json.dumps(
            parsed_output, ensure_ascii=False).encode('utf8'))
    else:
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)

        for parsed_file in parsed_output:
            title = Path(parsed_file["image_path"]).stem
            with open(Path("{}/{}.json".format(out_folder, title)), "w") as f:
                json.dump(parsed_file, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    # オプションの解析
    parser = argparse.ArgumentParser(
        description='Parse item drops from an F/GO screenshot.')
    # 3. parser.add_argumentで受け取る引数を追加していく
    parser.add_argument(
        '-i', '--filenames', help='image file to parse', nargs='+')    # 必須の引数を追加
    parser.add_argument(
        '-f', '--folder', help='folder containing images to parse')
    parser.add_argument('-o', '--out_folder',
                        help='folder to write parsed data to. If specified, parsed images will also be moved to here. Else, output will simply be written to stdout')
    parser.add_argument('-t', '--timeout', type=int, default=TIMEOUT,
                        help="images with the same amount of drops and QP are flagged as duplicate, if taken within this many seconds. Default: {}s".format(TIMEOUT))
    parser.add_argument('--ordering', help='sort files before processing. Needed to make use of missing screenshot detection',
                        type=Ordering, choices=list(Ordering), default=Ordering.NOTSPECIFIED)
    parser.add_argument(
        '-d', '--debug', help='output debug information', action='store_true')
    parser.add_argument('--version', action='version',
                        version=PROGNAME + " " + VERSION)
    parser.add_argument('-l', '--loglevel',
                        choices=('debug', 'info'), default='info')
    subparsers = parser.add_subparsers(
        title='subcommands', description='{subcommand} --help: show help message for the subcommand',)

    watcher_parser = subparsers.add_parser(
        'watch', help='continuously watch the folder specified by [-f FOLDER]')
    watcher_parser.add_argument(
        "-j",
        "--num_processes",
        required=False,
        default=DEFAULT_AMT_PROCESSES,
        type=int,
        help="number of processes to allocate in the process pool. Default: {}".format(
            DEFAULT_AMT_PROCESSES),
    )
    watcher_parser.add_argument(
        "-p",
        "--polling_frequency",
        required=False,
        default=DEFAULT_POLL_FREQ,
        type=int,
        help="how often to check for new images (in seconds). Default: {}s".format(
            DEFAULT_POLL_FREQ),
    )

    args = parser.parse_args()    # 引数を解析
    logging.basicConfig(
        level=logging.INFO,
        format='%(name)s <%(filename)s-L%(lineno)s> [%(levelname)s] %(message)s',
    )
    logger.setLevel(args.loglevel.upper())

    if args.out_folder is not None and not Path(args.out_folder):
        print("{} is not a valid path".format(args.out_folder))
        exit(1)

    for ndir in [Item_dir, CE_dir, Point_dir]:
        if not ndir.is_dir():
            ndir.mkdir(parents=True)

    # Attributes are only present if the watch subcommand has been invoked.
    if hasattr(args, "num_processes") and hasattr(args, "polling_frequency"):
        if args.folder is None or not Path(args.folder).exists():
            print(
                "The watch subcommands requires a valid input directory. Provide one with --folder.")
            exit(1)
        watch_parse_output_into_json(args)
    else:
        if args.filenames is None and args.folder is None:
            print(
                "No input files specified. Use --filenames or --folder to do so.")
            exit(1)
        # gather input image files
        if args.folder:
            inputs = [x for x in Path(args.folder).iterdir()]
        else:
            inputs = args.filenames

        inputs = sort_files(inputs, args.ordering)
        parsed_output = parse_into_json(inputs, args)
        output_json(parsed_output, args.out_folder)
