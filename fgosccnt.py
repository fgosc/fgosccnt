#!/usr/bin/env python3
import sys
import re
import argparse
from pathlib import Path
from collections import Counter
import csv
from enum import Enum
import itertools
import json
from operator import itemgetter
import math
import datetime
import logging
from typing import Tuple, Union
import io

import cv2
from numpy import ndarray
import numpy as np
import pytesseract
from PIL import Image
from PIL.ExifTags import TAGS

import pageinfo


PROGNAME = "FGOスクショカウント"
VERSION = "0.4.0"
DEFAULT_ITEM_LANG = "jpn"  # "jpn": japanese, "eng": English

logger = logging.getLogger(__name__)


class CustomAdapter(logging.LoggerAdapter):
    """
    この adapter を通した場合、自動的にログ出力文字列の先頭に [target] が挿入される。
    target は adapter インスタンス生成時に確定させること。
    """
    def process(self, msg, kwargs):
        return f"[{self.extra['target']}] {msg}", kwargs


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
train_item = basedir / Path("item.xml")  # item stack & bonus
train_chest = basedir / Path("chest.xml")  # drop_coount (Old UI)
train_dcnt = basedir / Path("dcnt.xml")  # drop_coount (New UI)
train_card = basedir / Path("card.xml")  # card name
drop_file = basedir / Path("fgoscdata/hash_drop.json")
eventquest_dir = basedir / Path("fgoscdata/data/json/")
items_img = basedir / Path("data/misc/items_img.png")
bunyan1_img = basedir / Path("data/misc/bunyan1.png")

hasher = cv2.img_hash.PHash_create()

FONTSIZE_UNDEFINED = -1
FONTSIZE_NORMAL = 0
FONTSIZE_SMALL = 1
FONTSIZE_TINY = 2
FONTSIZE_NEWSTYLE = 99
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
ID_FP = 4
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
ID_SYURENJYO_TMP = 94066100
ID_EVNET = 94000000
ID_GREEN_TEA = 94074504
ID_YELLOW_TEA = 94074505
ID_RED_TEA = 94074506
ID_WEST_AMERICA_AREA = 93040104
TIMEOUT = 15
QP_UNKNOWN = -1

class FgosccntError(Exception):
    pass


class GainedQPandDropMissMatchError(FgosccntError):
    pass


with open(drop_file, encoding='UTF-8') as f:
    drop_item = json.load(f)

# JSONファイルから各辞書を作成
item_name = {item["id"]: item["name"] for item in drop_item}
item_name_eng = {item["id"]: item["name_eng"] for item in drop_item
                 if "name_eng" in item.keys()}
item_shortname = {item["id"]: item["shortname"] for item in drop_item
                  if "shortname" in item.keys()}
item_dropPriority = {item["id"]: item["dropPriority"] for item in drop_item}
item_background = {item["id"]: item["background"] for item in drop_item
                   if "background" in item.keys()}
item_type = {item["id"]: item["type"] for item in drop_item}
dist_item = {item["phash_battle"]: item["id"] for item in drop_item
             if item["type"] == "Item" and "phash_battle" in item.keys()}
dist_ce = {item["phash"]: item["id"] for item in drop_item
           if item["type"] == "Craft Essence"}
dist_ce_narrow = {item["phash_narrow"]: item["id"] for item in drop_item
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
dist_exp_rarity["1fe03fe0517fa0bf"] = 9701200  # fix #368
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
    except (OSError, UnicodeEncodeError) as e:
        logger.exception(e)

npz = np.load(basedir / Path('background.npz'))
hist_zero = npz["hist_zero"]
hist_gold = npz["hist_gold"]
hist_silver = npz["hist_silver"]
hist_bronze = npz["hist_bronze"]


def has_intersect(a, b):
    """
    二つの矩形の当たり判定
    隣接するのはOKとする
    """
    return max(a[0], b[0]) < min(a[2], b[2]) \
        and max(a[1], b[1]) < min(a[3], b[3])


class State():
    def set_screen(self):
        self.screen_type = "normal"

    def set_char_position(self):
        logger.debug("JP Standard Position")

    def set_font_size(self):
        logger.debug("JP Standard Font Size")

    def set_max_qp(self):
        self.max_qp = 999999999
        logger.debug("999,999,999")


class JpNov2020(State):
    def set_screen(self):
        self.screen_type = "wide"


class JpAug2021(JpNov2020):
    def set_font_size(self):
        logger.debug("JP New Font Size")

    def set_max_qp(self):
        self.max_qp = 2000000000
        logger.debug("2,000,000,000")


class NaState(State):
    def set_char_position(self):
        logger.debug("NA Standard Position")


class NaOct2022(NaState):
    def set_screen(self):
        self.screen_type = "wide"

    def set_max_qp(self):
        self.max_qp = 2000000000
        logger.debug("2,000,000,000")


class Context:
    def __init__(self):
        self.jp_aug_2021 = JpAug2021()
        self.jp_nov_2020 = JpNov2020()
        self.jp = State()
        self.na = NaState()
        self.na_oct2022 = NaOct2022()
        self.state = self.jp_aug_2021
        self.set_screen()
        self.set_font_size()
        self.set_char_position()
        self.set_max_qp()

    def change_state(self, mode):
        if mode == "jp":
            self.state = self.jp_aug_2021
        elif mode == "na":
            self.state = self.na_oct2022
        else:
            raise ValueError("change_state method must be in {}".format(["jp", "na"]))
        self.set_screen()
        self.set_font_size()
        self.set_char_position()
        self.set_max_qp()

    def set_screen(self):
        self.state.set_screen()

    def set_char_position(self):
        self.state.set_char_position()

    def set_font_size(self):
        self.state.set_font_size()

    def set_max_qp(self):
        self.state.set_max_qp()


def get_coodinates(img: ndarray,
                   display: bool = False) -> Tuple[Tuple[int, int],
                                                   Tuple[int, int]]:
    threshold: int = 30

    height, width = img.shape[:2]
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if display:
        cv2.imshow('image', img_gray)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    _, inv = cv2.threshold(img_gray, threshold, 255, cv2.THRESH_BINARY_INV)
    if display:
        cv2.imshow('image', inv)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    contours, _ = cv2.findContours(inv, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours2 = []
    for cnt in contours:
        _, _, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if 1.81 < w/h < 1.83 and area > height / 2 * width / 2 and height/h > 1080/910:
            contours2.append(cnt)
    if len(contours2) == 0:
        raise ValueError("Game screen not found.")
    max_contour = max(contours2, key=lambda x: cv2.contourArea(x))
    x, y, width, height = cv2.boundingRect(max_contour)
    return ((x, y), (x + width, y + height))


def standardize_size(frame_img: ndarray,
                     display: bool = False) -> Tuple[ndarray, float]:
    TRAINING_WIDTH: int = 1754

    height, width = frame_img.shape[:2]
    if display:
        pass
    logger.debug("height: %d", height)
    logger.debug("width: %d", width)
    _, width, _ = frame_img.shape
    resize_scale: float = TRAINING_WIDTH / width
    logger.debug("resize_scale: %f", resize_scale)

    if resize_scale > 1:
        frame_img = cv2.resize(frame_img, (0, 0),
                               fx=resize_scale, fy=resize_scale,
                               interpolation=cv2.INTER_CUBIC)
    elif resize_scale < 1:
        frame_img = cv2.resize(frame_img, (0, 0),
                               fx=resize_scale, fy=resize_scale,
                               interpolation=cv2.INTER_AREA)
    if display:
        cv2.imshow('image', frame_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return frame_img, resize_scale


def area_decision(frame_img: ndarray,
                  display: bool = False) -> str:
    """
    FGOアプリの地域を選択
    "na", 'jp'に対応

    'items_img.png' とのオブジェクトマッチングで判定
    """
    img = frame_img[0:100, 0:500]
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if display:
        cv2.imshow('image', img_gray)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    template = imread(items_img, 0)
    res = cv2.matchTemplate(
                            img_gray,
                            template,
                            cv2.TM_CCOEFF_NORMED
                            )
    threshold = 0.9
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):
        return "na"
    return 'jp'


def check_page_mismatch(page_items: int, chestnum: int, pagenum: int, pages: int, lines: int) -> bool:
    if pages == 1:
        if chestnum + 1 != page_items:
            return False
        return True

    if not (pages - 1) * 21 <= chestnum <= pages * 21 - 1:
        return False
    if pagenum == pages:
        item_count = chestnum - ((pages - 1) * 21 - 1) + (pages * 3 - lines) * 7
        if item_count != page_items:
            return False
    return True


class ScreenShot:
    """
    戦利品スクリーンショットを表すクラス
    """

    def __init__(self, args, img_rgb, svm, svm_chest, svm_dcnt, svm_card,
                 fileextention, exLogger, reward_only=False):
        self.exLogger = exLogger
        threshold = 80
        self.img_rgb_orig = img_rgb
        img_blue, img_green, img_red = cv2.split(img_rgb)
        if (img_blue==img_green).all() & (img_green==img_red ).all():
            raise ValueError("Input image is grayscale")
        self.img_gray_orig = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        self.img_hsv_orig = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)
        _, self.img_th_orig = cv2.threshold(self.img_gray_orig,
                                            threshold, 255, cv2.THRESH_BINARY)

        ((self.x1, self.y1), (self.x2, self.y2)) = get_coodinates(self.img_rgb_orig)
        # Remove the extra notch by centering
        center = int((self.x2 - self.x1)/2 + self.x1)
        half_width = min(center, img_rgb.shape[1] - center)
        img_rgb_tmp = img_rgb[:, center - half_width:center + half_width]
        try:
            self.pagenum, self.pages, self.lines = pageinfo.guess_pageinfo(img_rgb_tmp)
            if self.lines / self.pages > 3:
                logger.warning("The maximum number of lines has been exceeded")
                self.lines = self.pages * 3
        except pageinfo.TooManyAreasDetectedError:
            self.pagenum, self.pages, self.lines = (-1, -1, -1)
        frame_img: ndarray = self.img_rgb_orig[self.y1: self.y2, self.x1: self.x2]
        img_resize, resize_scale = standardize_size(frame_img)
        self.img_rgb = img_resize
        mode = area_decision(img_resize)
        logger.debug("lang: %s", mode)
        # UI modeを決める
        sc = Context()
        sc.change_state(mode)
        self.max_qp = sc.state.max_qp
        self.screen_type = sc.state.screen_type
        dcnt_old, dcnt_new = self.drop_count_area(self.img_rgb_orig, resize_scale, sc)

        if logger.isEnabledFor(logging.DEBUG):
            cv2.imwrite('frame_img.png', img_resize)

        if logger.isEnabledFor(logging.DEBUG):
            if self.screen_type == "normal":
                cv2.imwrite('dcnt_old.png', dcnt_old)
            cv2.imwrite('dcnt_new.png', dcnt_new)

        self.img_gray = cv2.cvtColor(self.img_rgb, cv2.COLOR_BGR2GRAY)
        _, self.img_th = cv2.threshold(self.img_gray,
                                       threshold, 255, cv2.THRESH_BINARY)
        self.svm = svm
        self.svm_chest = svm_chest
        self.svm_dcnt = svm_dcnt

        self.height, self.width = self.img_rgb.shape[:2]
        if self.screen_type == "normal":
            self.chestnum = self.ocr_tresurechest(dcnt_old)
            if self.chestnum == -1:
                self.chestnum = self.ocr_dcnt(dcnt_new)
        else:
            self.chestnum = self.ocr_dcnt(dcnt_new)
        self.asr_y, self.actual_height = self.detect_scroll_bar()

        logger.debug("Total Drop (OCR): %d", self.chestnum)
        item_pts = self.img2points(mode)
        logger.debug("item_pts:%s", item_pts)

        self.items = []
        self.current_dropPriority = PRIORITY_REWARD_QP
        if reward_only:
            # qpsplit.py で利用
            item_pts = item_pts[0:1]
        prev_item = None

        # まんわか用イベント判定
        template1 = cv2.imread(str(bunyan1_img), 0)
        item15th = self.img_gray[item_pts[15][1]:item_pts[15][3], item_pts[15][0]:item_pts[15][2]]

        res = cv2.matchTemplate(item15th, template1, cv2.TM_CCOEFF_NORMED)
        threshold = 0.80
        loc = np.where(res >= threshold)
        self.Bunyan = False
        for pt in zip(*loc[::-1]):
            self.Bunyan = True
            break

        for i, pt in enumerate(item_pts):
            if self.Bunyan and i == 14:
                break
            lx, _ = self.find_edge(self.img_th[pt[1]: pt[3],
                                               pt[0]: pt[2]], reverse=True)
            logger.debug("lx: %d", lx)
            item_img_th = self.img_th[pt[1] + 37: pt[3] - 60,
                                      pt[0] + lx: pt[2] + lx]
            if self.is_empty_box(item_img_th):
                break
            item_img_rgb = self.img_rgb[pt[1]:  pt[3],
                                        pt[0] + lx:  pt[2] + lx]
            item_img_gray = self.img_gray[pt[1]: pt[3],
                                          pt[0] + lx: pt[2] + lx]
            if logger.isEnabledFor(logging.DEBUG):
                cv2.imwrite('item' + str(i) + '.png', item_img_rgb)
            dropitem = Item(args, i, prev_item, item_img_rgb, item_img_gray,
                            svm, svm_card, fileextention,
                            self.current_dropPriority, self.exLogger, mode)
            if dropitem.id == -1:
                break
            self.current_dropPriority = item_dropPriority[dropitem.id]
            if dropitem.id in [94069601, 94069602, 94069603]:
                # まんわかイベントのバニヤンに隠されているドロップが問題を生じるので補正
                dropitem.dropnum = 'x3'
            self.items.append(dropitem)
            prev_item = dropitem

        if self.Bunyan:
            lx, _ = self.find_edge(self.img_th[item_pts[14][1]: item_pts[14][3],
                                            item_pts[14][0]: item_pts[14][2]], reverse=True)
            item_img_rgb = self.img_rgb[item_pts[14][1]:  item_pts[14][3],
                                        item_pts[14][0] + lx:  item_pts[14][2] + lx]
            item_img_gray = self.img_gray[item_pts[14][1]: item_pts[14][3],
                                        item_pts[14][0] + lx: item_pts[14][2] + lx]
            dropitem = Item(args, i, prev_item, item_img_rgb, item_img_gray,
                            svm, svm_card, fileextention,
                            self.current_dropPriority, self.exLogger, mode)
            self.items.append(dropitem)

        self.itemlist = self.makeitemlist()
        try:
            self.total_qp = self.get_qp(mode)
            self.qp_gained = self.get_qp_gained(mode)
        except Exception as e:
            self.total_qp = -1
            self.qp_gained = -1
            self.exLogger.warning("QP detection fails")
            logger.exception(e)
        if self.qp_gained > 0 and len(self.itemlist) == 0:
            raise GainedQPandDropMissMatchError
        logger.debug(f'pagenum(pageninfo) pagenum: {self.pagenum}, pages: {self.pages}, lines: {self.lines}')
        self.pagenum, self.pages, self.lines = self.correct_pageinfo()
        logger.debug(f'pagenum(coreected) pagenum: {self.pagenum}, pages: {self.pages}, lines: {self.lines}')
        if not reward_only:
            self.check_page_mismatch()

    def check_page_mismatch(self):
        if self.Bunyan:
            num_items = len(self.itemlist) -1
        else:
            num_items = len(self.itemlist)
        valid = check_page_mismatch(
            num_items,
            self.chestnum,
            self.pagenum,
            self.pages,
            self.lines,
        )
        if not valid:
            self.exLogger.warning("drops_count is a mismatch:")
            self.exLogger.warning("drops_count = %d", self.chestnum)
            self.exLogger.warning("drops_found = %d", len(self.itemlist))

    def find_notch(self):
        """
        直線検出で検出されなかったフチ幅を検出
        """
        edge_width = 200

        height, width = self.img_hsv_orig.shape[:2]
        target_color = 0
        for lx in range(edge_width):
            img_hsv_x = self.img_hsv_orig[:, lx: lx + 1]
            # ヒストグラムを計算
            hist = cv2.calcHist([img_hsv_x], [0], None, [256], [0, 256])
            # 最小値・最大値・最小値の位置・最大値の位置を取得
            _, maxVal, _, maxLoc = cv2.minMaxLoc(hist)
            if not (maxLoc[1] == target_color and maxVal > height * 0.7):
                break

        for rx in range(edge_width):
            img_hsv_x = self.img_hsv_orig[:, width - rx - 1: width - rx]
            # ヒストグラムを計算
            hist = cv2.calcHist([img_hsv_x], [0], None, [256], [0, 256])
            # 最小値・最大値・最小値の位置・最大値の位置を取得
            _, maxVal, _, maxLoc = cv2.minMaxLoc(hist)
            if not (maxLoc[1] == target_color and maxVal > height * 0.7):
                break

        return lx, rx

    def drop_count_area(self, img: ndarray,
                        resize_scale,
                        sc,
                        display: bool = False) -> Tuple[Union[ndarray, None], ndarray]:
        # widescreenかどうかで挙動を変える
        if resize_scale > 1:
            img = cv2.resize(img, (0, 0),
                             fx=resize_scale, fy=resize_scale,
                             interpolation=cv2.INTER_CUBIC)
        elif resize_scale < 1:
            img = cv2.resize(img, (0, 0),
                             fx=resize_scale, fy=resize_scale,
                             interpolation=cv2.INTER_AREA)
        # ((x1, y1), (_, _)) = get_coodinates(img)
        # 相対座標(旧UI)
        dcnt_old = None
        if sc.state.screen_type == "normal":
            dcnt_old = img[int(self.y1*resize_scale) - 81: int(self.y1*resize_scale) - 44,
                           int(self.x1*resize_scale) + 1446: int(self.x1*resize_scale) + 1505]
            if display:
                cv2.imshow('image', dcnt_old)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        # 相対座標(新UI)
        lx, rx = self.find_notch()
        height, width = img.shape[:2]
        if (width - lx - rx)/height > 16/8.96:  # Issue #317
            # Widescreen
            dcnt_new = img[int(self.y1*resize_scale) - 20: int(self.y1*resize_scale) + 13,
                           width - 495 - rx: width - 415 - int(rx*resize_scale)]
        else:
            dcnt_new = img[int(self.y1*resize_scale) - 20: int(self.y1*resize_scale) + 13,
                           width - 430 : width - 340]
        if display:
            cv2.imshow('image', dcnt_new)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return dcnt_old, dcnt_new

    def detect_scroll_bar(self):
        '''
        Modified from determine_scroll_position()
        '''
        width = self.img_rgb.shape[1]
        topleft = (width - 90, 81)
        bottomright = (width, 2 + 753)

        if logger.isEnabledFor(logging.DEBUG):
            img_copy = self.img_rgb.copy()
            cv2.rectangle(img_copy, topleft, bottomright, (0, 0, 255), 3)
            cv2.imwrite("./scroll_bar_selected2.jpg", img_copy)

        gray_image = self.img_gray[
                                   topleft[1]: bottomright[1],
                                   topleft[0]: bottomright[0]
                                   ]
        _, binary = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)
        if logger.isEnabledFor(logging.DEBUG):
            cv2.imwrite("scroll_bar_binary2.png", binary)
        contours = cv2.findContours(
                                    binary,
                                    cv2.RETR_LIST,
                                    cv2.CHAIN_APPROX_NONE
                                    )[0]
        pts = []
        for cnt in contours:
            ret = cv2.boundingRect(cnt)
            pt = [ret[0], ret[1], ret[0] + ret[2], ret[1] + ret[3]]
            if ret[3] > 10:
                pts.append(pt)
        if len(pts) == 0:
            logger.debug("Can't find scroll bar")
            return -1, -1
        elif len(pts) > 1:
            max_cnt = max(contours, key=lambda x: cv2.contourArea(x))
            ret = cv2.boundingRect(max_cnt)
            pt = [ret[0], ret[1], ret[0] + ret[2], ret[1] + ret[3]]
            if ret[3] <= 10:
                logger.debug("Can't find scroll bar")
                return -1, -1
        return pt[1], pt[3] - pt[1]


    def valid_pageinfo(self):
        '''
        Checking the content of pageinfo and correcting it when it fails
        '''
        if self.pagenum == -1 or self.pages == -1 or self.lines == -1:
            return False
        if (self.pagenum == 1 and self.pages == 1 and self.lines == 0) and self.chestnum > 20:
            return False
        elif self.itemlist[0]["id"] != ID_REWARD_QP and self.pagenum == 1:
            return False
        elif self.Bunyan and self.chestnum != -1 and self.pagenum != 1 \
                and self.lines != int(self.chestnum/7) + 2:
            return False
        elif self.Bunyan is False and self.chestnum != -1 and self.pagenum != 1 \
                and self.lines != int(self.chestnum/7) + 1:
            return False
        return True

    def correct_pageinfo(self):
        if self.valid_pageinfo() is False:
            self.exLogger.warning("pageinfo validation failed")
            if self.asr_y == -1 or self.actual_height == -1:
                return 1, 1, 0
            entire_height = 645
            esr_y = 17
            cap_height = 14  # 正規化後の im.height を 1155 であると仮定して計算した値
            pagenum = pageinfo.guess_pagenum(self.asr_y, esr_y, self.actual_height, entire_height, cap_height)
            pages = pageinfo.guess_pages(self.actual_height, entire_height, cap_height)
            lines = pageinfo.guess_lines(self.actual_height, entire_height, cap_height)
            return pagenum, pages, lines
        else:
            return self.pagenum, self.pages, self.lines

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
            if self.screen_type == "normal":
                pt = ((288, 948), (838, 1024))
            else:
                pt = ((288, 838), (838, 914))
            logger.debug('Use tesseract')
            qp_total_text = self.extract_text_from_image(
                self.img_rgb[pt[0][1]: pt[1][1], pt[0][0]: pt[1][0]]
            )
            logger.debug('qp_total_text from text: %s', qp_total_text)
            qp_total = self.get_qp_from_text(qp_total_text)

        logger.debug('qp_total from text: %s', qp_total)
        if qp_total > self.max_qp:
            self.exLogger.warning(
                "qp_total exceeds the system's maximum: %s", qp_total
            )
        if qp_total == 0:
            return QP_UNKNOWN

        return qp_total

    def get_qp_gained(self, mode):
        use_tesseract = False
        bounds = pageinfo.detect_qp_region(self.img_rgb_orig, mode)
        if bounds is None:
            # fall back on hardcoded bound
            if self.screen_type == "normal":
                bounds = ((398, 858), (948, 934))
            else:
                bounds = ((398, 748), (948, 824))
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
        if logger.isEnabledFor(logging.DEBUG):
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
            img_th_x = img_th[:, width - j - 1: width - j]
            # ヒストグラムを計算
            hist = cv2.calcHist([img_th_x], [0], None, [256], [0, 256])
            # 最小値・最大値・最小値の位置・最大値の位置を取得
            _, _, _, maxLoc = cv2.minMaxLoc(hist)
            if maxLoc[1] == 0:
                break
        rx = j

        return lx, rx

    def makeitemlist(self):
        """
        アイテムを出力
        """
        itemlist = []
        for item in self.items:
            tmp = {}
            tmp['id'] = item.id
            tmp['name'] = item.name
            tmp['dropPriority'] = item_dropPriority[item.id]
            tmp['dropnum'] = int(item.dropnum[1:])
            tmp['bonus'] = item.bonus
            tmp['category'] = item.category
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
                    and 0.3 < ret[2]/ret[3] < 0.85 and ret[3] > h * 0.45:
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
        if len(item_pts) > len(str(self.max_qp)):
            # QP may be misrecognizing the 10th digit or more, so cut it
            item_pts = item_pts[len(item_pts) - len(str(self.max_qp)):]
        logger.debug("ocr item_pts: %s", item_pts)
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

            if pt[0] == 0:
                tmpimg = im_th[pt[1]:pt[3], pt[0]:pt[2]+1]
            else:
                tmpimg = im_th[pt[1]:pt[3], pt[0]-1:pt[2]+1]
            tmpimg = cv2.resize(tmpimg, (win_size))
            hog = cv2.HOGDescriptor(win_size, block_size,
                                    block_stride, cell_size, bins)
            test.append(hog.compute(tmpimg))  # 特徴量の格納
            test = np.array(test)

            pred = self.svm_chest.predict(test)
            res = res + str(int(pred[1][0][0]))

        return int(res)

    def ocr_tresurechest(self, drop_count_img):
        """
        宝箱数をOCRする関数
        """
        threshold = 80
        img_gray = cv2.cvtColor(drop_count_img, cv2.COLOR_BGR2GRAY)
        _, img_num = cv2.threshold(img_gray,
                                   threshold, 255, cv2.THRESH_BINARY)
        im_th = cv2.bitwise_not(img_num)
        h, w = im_th.shape[:2]

        # 情報ウィンドウが数字とかぶった部分を除去する
        for y in range(h):
            im_th[y, 0] = 255
        for x in range(w):  # ドロップ数7のときバグる対策 #54
            im_th[0, x] = 255
        return self.ocr_text(im_th)

    def pred_dcnt(self, img):
        """
        for JP new UI
        """
        # Hog特徴のパラメータ
        win_size = (120, 60)
        block_size = (16, 16)
        block_stride = (4, 4)
        cell_size = (4, 4)
        bins = 9
        char = []

        tmpimg = cv2.resize(img, (win_size))
        hog = cv2.HOGDescriptor(win_size, block_size,
                                block_stride, cell_size, bins)
        char.append(hog.compute(tmpimg))  # 特徴量の格納
        char = np.array(char)

        pred = self.svm_dcnt.predict(char)
        res = str(int(pred[1][0][0]))

        return int(res)

    def img2num(self, img, img_th, pts, char_w, end):
        """実際より小さく切り抜かれた数字画像を補正して認識させる

        """
        height, width = img.shape[:2]
        c_center = int(pts[0] + (pts[2] - pts[0])/2)
        # newimg = img[:, item_pts[-1][0]-1:item_pts[-1][2]+1]
        newimg = img[:, max(int(c_center - char_w/2), 0):min(int(c_center + char_w/2), width)]

        threshold2 = 10
        ret, newimg_th = cv2.threshold(newimg,
                                       threshold2,
                                       255,
                                       cv2.THRESH_BINARY)
        # 上部はもとのやつを上書き
        # for w in range(item_pts[-1][2] - item_pts[-1][0] + 2):
        for w in range(min(int(c_center + char_w/2), width) - max(int(c_center - char_w/2), 0)):
            for h in range(end):
                newimg_th[h, w] = img_th[h, w + int(c_center - char_w/2)]
        #        newimg_th[h, w] = img_th[h, w + item_pts[-1][0]]
            newimg_th[height - 1, w] = 0
            newimg_th[height - 2, w] = 0
            newimg_th[height - 3, w] = 0

        res = self.pred_dcnt(newimg_th)
        return res

    def ocr_dcnt(self, drop_count_img):
        """
        ocr drop_count (for New UI)
        """
        char_w = 28
        threshold = 80
        kernel = np.ones((4, 4), np.uint8)
        img = cv2.cvtColor(drop_count_img, cv2.COLOR_BGR2GRAY)
        _, img_th = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
        img_th = cv2.dilate(img_th, kernel, iterations=1)
        height, width = img_th.shape[:2]

        end = -1
        for i in range(height):
            if end == -1 and img_th[height - i - 1, width - 1] == 255:
                end = height - i
                break
        start = end - 7

        for j in range(width):
            for k in range(end - start):
                img_th[start + k, j] = 0

        contours = cv2.findContours(img_th,
                                    cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)[0]
        item_pts = []
        for cnt in contours:
            ret = cv2.boundingRect(cnt)
            pt = [ret[0], ret[1], ret[0] + ret[2], ret[1] + ret[3]]
            if ret[1] > 0 and ret[3] > 8 and ret[1] + ret[3] == start \
               and 12 < ret[2] < char_w + 4 and ret[0] + ret[2] != width:
                item_pts.append(pt)

        if len(item_pts) == 0:
            return -1
        item_pts.sort()

        res = self.img2num(img, img_th, item_pts[-1], char_w, end)
        if len(item_pts) >= 2:
            if item_pts[-1][0] - item_pts[-2][2] < char_w / (2 / 3):
                res2 = self.img2num(img, img_th, item_pts[-2], char_w, end)
                res = res2 * 10 + res
                if len(item_pts) == 3:
                    if item_pts[-2][0] - item_pts[-3][2] < char_w / (2 / 3):
                        res3 = self.img2num(img, img_th, item_pts[-3], char_w, end)
                        res = res3 * 100 + res2 * 10 + res

        return res

    def calc_offset(self, pts, std_pts, margin_x):
        """
        オフセットを反映
        """
        scroll_limit_uppler = 110
        scroll_limit_lower = 285

        if len(pts) == 0:
            return std_pts
        # Y列でソート
        pts.sort(key=lambda x: x[1])
        if scroll_limit_uppler < pts[0][1] < scroll_limit_lower:
            raise ValueError("Incorrect scroll position.")

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

    def img2points(self, mode):
        """
        戦利品左一列のY座標を求めて標準座標とのずれを補正して座標を出力する
        """
        height_limit_first_item = 87
        std_pts = self.booty_pts()

        row_size = 7  # アイテム表示最大列
        col_size = 3  # アイテム表示最大行
        margin_x = 15
        area_size_lower = 37000  # アイテム枠の面積の最小値
        img_1strow = self.img_th[0:self.height,
                                 std_pts[0][0] - margin_x:
                                 std_pts[0][2] + margin_x]

        SCROLLBAR_WIDTH4ONEPAGE = 610
        POSITION_TOP = 16
        POSITION_BOTTOM_JP = 42  # JP
        POSITION_BOTTOM_NA = 52  # NA
        SCROLL_OFFSET = 28

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
                if 4 <= len(approx) <= 6:  # 六角形のみ認識
                    ret = cv2.boundingRect(cnt)
                    if ret[1] > height_limit_first_item \
                       and ret[1] + ret[3] < self.height * 0.76 - 101:
                        # 小数の数値はだいたいの実測
                        pts = [ret[0], ret[1],
                               ret[0] + ret[2], ret[1] + ret[3]]
                        leftcell_pts.append(pts)
        if len(leftcell_pts) != 0:
            item_pts = self.calc_offset(leftcell_pts, std_pts, margin_x)
            logger.debug("leftcell_pts: %s", leftcell_pts)
        else:
            if (self.asr_y == POSITION_TOP and self.actual_height == SCROLLBAR_WIDTH4ONEPAGE) or self.actual_height == -1:
                # case: normal
                if mode == "na":
                    leftcell_pts = [[25, 109, 214, 315]]
                else:
                    leftcell_pts = [[14, 97, 202, 303]]
                item_pts = self.calc_offset(leftcell_pts, std_pts, margin_x)
            elif POSITION_BOTTOM_JP <= self.asr_y <= POSITION_BOTTOM_NA and self.actual_height == SCROLLBAR_WIDTH4ONEPAGE:
                # case: scrolling down by mistake
                if mode == "na":
                    leftcell_pts = [[25, 299, 214, 504]]
                else:
                    leftcell_pts = [[14, 97 - SCROLL_OFFSET, 202, 303 - SCROLL_OFFSET]]
                item_pts = self.calc_offset(leftcell_pts, std_pts, margin_x)

        return item_pts

    def booty_pts(self):
        """
        戦利品が出現する21の座標 [left, top, right, bottom]
        解像度別に設定
        """
        criteria_left = 102
        criteria_top = 99
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
    def __init__(self, args, pos, prev_item, img_rgb, img_gray, svm, svm_card,
                 fileextention, current_dropPriority, exLogger, mode='jp'):
        self.position = pos
        self.prev_item = prev_item
        self.img_rgb = img_rgb
        self.img_gray = img_gray
        self.img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)
        _, img_th = cv2.threshold(self.img_gray, 174, 255, cv2.THRESH_BINARY)
        self.img_th = cv2.bitwise_not(img_th)
        self.fileextention = fileextention
        self.exLogger = exLogger
        self.dropnum_cache = []
        self.margin_left = 5

        self.height, self.width = img_rgb.shape[:2]
        logger.debug("pos: %d", pos)
        self.identify_item(args, prev_item, svm_card,
                           current_dropPriority)
        if self.id == -1:
            return
        logger.debug("id: %d", self.id)
        logger.debug("background: %s", self.background)
        logger.debug("dropPriority: %s", item_dropPriority[self.id])
        logger.debug("Category: %s", self.category)
        logger.debug("Name: %s", self.name)

        self.svm = svm
        self.bonus = ""
        if self.category != "Craft Essence" and self.category != "Exp. UP":
            self.ocr_digit(mode)
        else:
            self.dropnum = "x1"
        logger.debug("Bonus: %s", self.bonus)
        logger.debug("Stack: %s", self.dropnum)

    def identify_item(self, args, prev_item, svm_card,
                      current_dropPriority):
        self.background = classify_background(self.img_rgb)
        self.hash_item = compute_hash(self.img_rgb)  # 画像の距離
        if prev_item is not None:
            # [Requirements for Caching]
            # 1. previous item is not a reward QP.
            # 2. Same background as the previous item
            # 3. Not (similarity is close) dice, gem or EXP
            if prev_item.id != ID_REWARD_QP \
                and prev_item.background == self.background \
                and not (ID_GEM_MIN <= prev_item.id <= ID_SECRET_GEM_MAX or
                         ID_2ZORO_DICE <= prev_item.id <= ID_3ZORO_DICE or
                         ID_EXP_MIN <= prev_item.id <= ID_EXP_MAX or
                         ID_GREEN_TEA <= prev_item.id <= ID_RED_TEA):
                d = hasher.compare(self.hash_item, prev_item.hash_item)
                if d <= 4:
                    self.category = prev_item.category
                    self.id = prev_item.id
                    self.name = prev_item.name
                    return
        self.category = self.classify_category(svm_card)
        self.id = self.classify_card(self.img_rgb, current_dropPriority)
        if args.lang == "jpn":
            self.name = item_name[self.id]
        else:
            if self.id in item_name_eng.keys():
                self.name = item_name_eng[self.id]
            else:
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

    def detect_bonus_char4jpg(self, mode="jp"):
        """
        [JP]Ver.2.37.0以前の仕様
        戦利品数OCRで下段の黄文字の座標を抽出する
        PNGではない画像の認識用

        """
        # QP,ポイントはボーナス6桁のときに高さが変わる
        # それ以外は3桁のときに変わるはず(未確認)
        # ここのmargin_right はドロップ数の下一桁目までの距離
        # base_line = 181 if mode == "na" else 179
        base_line = 179
        pattern_tiny = r"^\(\+\d{4,5}0\)$"
        pattern_small = r"^\(\+\d{5}0\)$"
        pattern_normal = r"^\(\+[1-9]\d*\)$"
        # 1-5桁の読み込み
        font_size = FONTSIZE_NORMAL
        # if mode == "na":
        #     margin_right = 20
        # else:
        #     margin_right = 26
        margin_right = 26
        line, pts = self.get_number4jpg(base_line, margin_right, font_size, mode)
        logger.debug("Read BONUS NORMAL: %s", line)
        m_normal = re.match(pattern_normal, line)
        if m_normal:
            logger.debug("Font Size: %d", font_size)
            return line, pts, font_size
        # 6桁の読み込み
        # if mode == "na":
        #     margin_right = 19
        # else:
        #     margin_right = 25
        margin_right = 25
        font_size = FONTSIZE_SMALL
        line, pts = self.get_number4jpg(base_line, margin_right, font_size, mode)
        logger.debug("Read BONUS SMALL: %s", line)
        m_small = re.match(pattern_small, line)
        if m_small:
            logger.debug("Font Size: %d", font_size)
            return line, pts, font_size
        # 7桁読み込み
        font_size = FONTSIZE_TINY
        # if mode == "na":
        #     margin_right = 18
        # else:
        #     margin_right = 26
        margin_right = 26
        line, pts = self.get_number4jpg(base_line, margin_right, font_size, mode)
        logger.debug("Read BONUS TINY: %s", line)
        m_tiny = re.match(pattern_tiny, line)
        if m_tiny:
            logger.debug("Font Size: %d", font_size)
            return line, pts, font_size
        else:
            font_size = FONTSIZE_UNDEFINED
            logger.debug("Font Size: %d", font_size)
            line = ""
            pts = []

        return line, pts, font_size

    def detect_bonus_char4jpg2(self, mode):
        """
        [JP]Ver.2.37.0以降の仕様
        戦利品数OCRで下段の黄文字の座標を抽出する
        PNGではない画像の認識用

        """
        # QP,ポイントはボーナス6桁のときに高さが変わる
        # それ以外は3桁のときに変わるはず(未確認)
        # ここのmargin_right はドロップ数の下一桁目までの距離
        # base_line = 181 if mode == "na" else 179
        base_line = 179
        pattern_tiny = r"^\(\+\d{4,5}0\)$"
        pattern_small = r"^\(\+\d{5}0\)$"
        pattern_normal = r"^\(\+[1-9]\d*\)$"
        font_size = FONTSIZE_NEWSTYLE
        # if mode == "na":
        #     margin_right = 20
        # else:
        #     margin_right = 26
        margin_right = 26
        # 1-5桁の読み込み
        cut_width = 21
        comma_width = 5
        line, pts = self.get_number4jpg2(base_line, margin_right, cut_width, comma_width)
        logger.debug("Read BONUS NORMAL: %s", line)
        m_normal = re.match(pattern_normal, line)
        if m_normal:
            logger.debug("Font Size: %d", font_size)
            return line, pts, font_size
        # 6桁の読み込み
        cut_width = 19
        comma_width = 5

        line, pts = self.get_number4jpg2(base_line, margin_right, cut_width, comma_width)
        logger.debug("Read BONUS SMALL: %s", line)
        m_small = re.match(pattern_small, line)
        if m_small:
            logger.debug("Font Size: %d", font_size)
            return line, pts, font_size
        # 7桁読み込み
        cut_width = 18
        comma_width = 5

        line, pts = self.get_number4jpg2(base_line, margin_right, cut_width, comma_width)
        logger.debug("Read BONUS TINY: %s", line)
        m_tiny = re.match(pattern_tiny, line)
        if m_tiny:
            logger.debug("Font Size: %d", font_size)
            return line, pts, font_size
        else:
            font_size = FONTSIZE_UNDEFINED
            logger.debug("Font Size: %d", font_size)
            line = ""
            pts = []

        return line, pts, font_size

    def define_fontsize(self, font_size, mode="jp"):
        # if mode == "jp":
        #     if font_size == FONTSIZE_NORMAL:
        #         cut_width = 20
        #         cut_height = 28
        #         comma_width = 9
        #     elif font_size == FONTSIZE_SMALL:
        #         cut_width = 18
        #         cut_height = 25
        #         comma_width = 8
        #     else:
        #         cut_width = 16
        #         cut_height = 22
        #         comma_width = 6
        # else:
        #     if font_size == FONTSIZE_NORMAL:
        #         cut_width = 18
        #         cut_height = 27
        #         comma_width = 8
        #     elif font_size == FONTSIZE_SMALL:
        #         cut_width = 18
        #         cut_height = 25
        #         comma_width = 8
        #     else:
        #         cut_width = 16
        #         cut_height = 22
        #         comma_width = 6
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

    def get_number4jpg(self, base_line, margin_right, font_size, mode="jp"):
        """[JP]Ver.2.37.0以前の仕様
        """
        cut_width, cut_height, comma_width = self.define_fontsize(font_size, mode)
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

    def get_number4jpg2(self, base_line, margin_right, cut_width, comma_width):
        """[JP]Ver.2.37.0以降の仕様

        """
        cut_height = 30
        top_y = base_line - cut_height
        # まず、+, xの位置が何桁目か調査する
        pts = []
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

    def get_number(self, base_line, margin_right, font_size, mode):
        """[JP]Ver.2.37.0以前の仕様
        """
        cut_width, cut_height, comma_width = self.define_fontsize(font_size, mode)
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
                self.margin_left = pt[0]
                break
        # 決まった位置まで出力する
        line = ""
        for j in range(i):
            if (self.name == "QP" or self.category in ["Point"]) and j < 2:
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

    def get_number2(self, cut_width, comma_width, base_line=147, margin_right=15):
        """[JP]Ver.2.37.0以降の仕様
        """
        cut_height = 26
        # base_line = 147
        # margin_right = 15
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
                self.margin_left = pt[0]
                break
        # 決まった位置まで出力する
        line = ""
        for j in range(i):
            if (self.name == "QP" or self.category in ["Point"]) and j < 2:
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

    def detect_white_char(self, base_line, margin_right, mode="jp"):
        """
        上段と下段の白文字を見つける機能を一つに統合
        [JP]Ver.2.37.0からボーナスがある場合の表示の仕様変更有り
        """
        pattern_tiny = r"^[\+x][1-9]\d{4}00$"
        pattern_tiny_qp = r"^\+[1-9]\d{4,5}00$"
        pattern_small = r"^[\+x]\d{4}00$"
        pattern_small_qp = r"^\+\d{4,5}00$"
        pattern_normal = r"^[\+x][1-9]\d{0,5}$"
        pattern_normal_qp = r"^\+[1-9]\d{0,4}0$"
        logger.debug("base_line: %d", base_line)
#        if mode == "jp" and base_line < 170:
        if base_line < 170:
            # JP Ver.2.37.0以降の新仕様
            # 1-6桁の読み込み
            font_size = FONTSIZE_NEWSTYLE
            cut_width = 21
            comma_width = 5
            line = self.get_number2(cut_width, comma_width)
            logger.debug("Read NORMAL: %s", line)
            if self.id == ID_QP or self.category == "Point":
                pattern_normal = pattern_normal_qp
            m_normal = re.match(pattern_normal, line)
            if m_normal:
                logger.debug("Font Size: %d", font_size)
                self.font_size = font_size
                return line
            # 6桁の読み込み
            cut_width = 19
            comma_width = 5
            line = self.get_number2(cut_width, comma_width)
            logger.debug("Read SMALL: %s", line)
            if self.id == ID_QP or self.category == "Point":
                pattern_small = pattern_small_qp
            m_small = re.match(pattern_small, line)
            if m_small:
                logger.debug("Font Size: %d", font_size)
                self.font_size = font_size
                return line
            # 7桁読み込み
            cut_width = 19
            comma_width = 4
            line = self.get_number2(cut_width, comma_width)
            logger.debug("Read TINY: %s", line)
            if self.id == ID_QP or self.category == "Point":
                pattern_tiny = pattern_tiny_qp
            m_tiny = re.match(pattern_tiny, line)
            if m_tiny:
                logger.debug("Font Size: %d", font_size)
                self.font_size = font_size
                return line
        # elif mode == "jp" and self.id not in [ID_QP, ID_REWARD_QP] and self.category != "Point":
        elif self.id not in [ID_QP, ID_REWARD_QP] and self.category != "Point":
            cut_width = 21
            comma_width = 5
            line = self.get_number2(cut_width, comma_width, base_line=base_line, margin_right=margin_right)
            logger.debug("line: %s", line)
            if len(line) <= 1:
                return ""
            elif not line[1:].isdigit():
                return ""
            return line
        else:
            # JP Ver.2.37.0以前の旧仕様
            if self.font_size != FONTSIZE_UNDEFINED:
                line = self.get_number(base_line, margin_right, self.font_size, mode)
                logger.debug("line: %s", line)
                if len(line) <= 1:
                    return ""
                elif not line[1:].isdigit():
                    return ""
                return line
            else:
                # 1-6桁の読み込み
                font_size = FONTSIZE_NORMAL
                line = self.get_number(base_line, margin_right, font_size, mode)
                logger.debug("Read NORMAL: %s", line)
                if self.id == ID_QP or self.category == "Point":
                    pattern_normal = pattern_normal_qp
                m_normal = re.match(pattern_normal, line)
                if m_normal:
                    logger.debug("Font Size: %d", font_size)
                    self.font_size = font_size
                    return line
                # 6桁の読み込み
                font_size = FONTSIZE_SMALL
                line = self.get_number(base_line, margin_right, font_size, mode)
                logger.debug("Read SMALL: %s", line)
                if self.id == ID_QP or self.category == "Point":
                    pattern_small = pattern_small_qp
                m_small = re.match(pattern_small, line)
                if m_small:
                    logger.debug("Font Size: %d", font_size)
                    self.font_size = font_size
                    return line
                # 7桁読み込み
                font_size = FONTSIZE_TINY
                line = self.get_number(base_line, margin_right, font_size, mode)
                logger.debug("Read TINY: %s", line)
                if self.id == ID_QP or self.category == "Point":
                    pattern_tiny = pattern_tiny_qp
                m_tiny = re.match(pattern_tiny, line)
                if m_tiny:
                    logger.debug("Font Size: %d", font_size)
                    self.font_size = font_size
                    return line
        return ""

    def read_item(self, pts):
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
        logger.debug("OCR Result: %s", lines)
        # 以下エラー訂正
        if not lines.endswith(")"):
            lines = lines[:-1] + ")"
        if not lines.startswith("(+") and not lines.startswith("(x"):
            if lines[0] in ["+", 'x']:
                lines = "(" + lines
            elif lines.startswith("("):
                lines = lines.replace("(", "(+")
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
            elif self.name == "QP" or self.name == "クエストクリア報酬QP" or self.name == "フレンドポイント":
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

    def ocr_digit(self, mode='jp'):
        """
        戦利品OCR
        """
        self.font_size = FONTSIZE_UNDEFINED

        if self.prev_item is None:
            prev_id = -1
        else:
            prev_id = self.prev_item.id

        logger.debug("self.id: %d", self.id)
        logger.debug("prev_id: %d", prev_id)
        if prev_id == self.id:
            self.dropnum_cache = self.prev_item.dropnum_cache
        if prev_id == self.id \
                and not (ID_GEM_MAX <= self.id <= ID_MONUMENT_MAX) and not (ID_GREEN_TEA <= self.id <= ID_RED_TEA):
            # もしキャッシュ画像と一致したらOCRスキップ
            # logger.debug("dropnum_cache: %s", self.prev_item.dropnum_cache)
            for dropnum_cache in self.prev_item.dropnum_cache:
                pts = dropnum_cache["pts"]
                img_gray = self.img_gray[pts[0][1]-2:pts[1][1]+2,
                                         pts[0][0]-2:pts[1][0]+2]
                template = dropnum_cache["img"]
                res = cv2.matchTemplate(img_gray, template,
                                        cv2.TM_CCOEFF_NORMED)
                threshold = 0.97
                loc = np.where(res >= threshold)
                find_match = False
                for pt in zip(*loc[::-1]):
                    find_match = True
                    break
                if find_match:
                    logger.debug("find_match")
                    self.bonus = dropnum_cache["bonus"]
                    self.dropnum = dropnum_cache["dropnum"]
                    self.bonus_pts = dropnum_cache["bonus_pts"]
                    return
            logger.debug("not find_match")

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
        else:
            self.bonus, self.bonus_pts, self.font_size = self.detect_bonus_char4jpg2(mode)
            # if mode == "jp":
            #     self.bonus, self.bonus_pts, self.font_size = self.detect_bonus_char4jpg2(mode)
            # else:
            #     self.bonus, self.bonus_pts, self.font_size = self.detect_bonus_char4jpg(mode)
        logger.debug("Bonus Font Size: %s", self.font_size)

        # 実際の(ボーナス無し)ドロップ数が上段にあるか下段にあるか決定
        offset_y = 0
        if (self.category in ["Quest Reward", "Point"] or self.name == "QP") \
           and len(self.bonus) >= 5:  # ボーナスは"(+*0)"なので
            # 1桁目の上部からの距離を設定
            base_line = self.bonus_pts[-2][1] - 3 + offset_y
        else:
            base_line = int(180/206*self.height)

        # 実際の(ボーナス無し)ドロップ数の右端の位置を決定
        # offset_x = -7 if mode == "na" else 0
        offset_x = 0
        if self.category in ["Quest Reward", "Point"] or self.name == "QP":
            margin_right = 15 + offset_x
        elif len(self.bonus_pts) > 0:
            margin_right = self.width - self.bonus_pts[0][0] + 2
        else:
            margin_right = 15 + offset_x
        logger.debug("margin_right: %d", margin_right)
        self.dropnum = self.detect_white_char(base_line, margin_right, mode)
        logger.debug("self.dropnum: %s", self.dropnum)
        if len(self.dropnum) == 0:
            self.dropnum = "x1"
        if self.id != ID_REWARD_QP \
                and not (ID_GEM_MAX <= self.id <= ID_MONUMENT_MAX):
            dropnum_found = False
            for cache_item in self.dropnum_cache:
                if cache_item["dropnum"] == self.dropnum:
                    dropnum_found = True
                    break
            if dropnum_found is False:
                # キャッシュのために画像を取得する
                _, width = self.img_gray.shape
                _, cut_height, _ = self.define_fontsize(self.font_size, mode)
                logger.debug("base_line: %d", base_line)
                logger.debug("cut_height: %d", cut_height)
                logger.debug("margin_right: %d", margin_right)
                pts = ((self.margin_left, base_line - cut_height),
                       (width - margin_right, base_line))
                cached_img = self.img_gray[pts[0][1]:pts[1][1],
                                           pts[0][0]:pts[1][0]]
                tmp = {}
                tmp["dropnum"] = self.dropnum
                tmp["img"] = cached_img
                tmp["pts"] = pts
                tmp["bonus"] = self.bonus
                tmp["bonus_pts"] = self.bonus_pts
                self.dropnum_cache.append(tmp)

    def gem_img2id(self, img, gem_dict):
        hash_gem = self.compute_gem_hash(img)
        gems = {}
        for i in gem_dict.keys():
            d2 = hasher.compare(hash_gem, hex2hash(gem_dict[i]))
            if d2 <= 21:
                gems[i] = d2
        gems = sorted(gems.items(), key=lambda x: x[1])
        gem = next(iter(gems))
        return gem[0]

    def classify_tea(self, img):
        def calc_hist(img):
            # ヒストグラムを計算する。
            hist = cv2.calcHist([img], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
            # ヒストグラムを正規化する。
            hist = cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
            # (n_bins, 1) -> (n_bins,)
            hist = hist.squeeze(axis=-1)

            return hist

        def calc_hue_hist(img):
            # HSV 形式に変換する。
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # 成分ごとに分離する。
            h, s, v = cv2.split(hsv)
            # Hue 成分のヒストグラムを計算する。
            hist = calc_hist(h)

            return hist

        red_tea_file = Path(__file__).resolve().parent / 'data/misc/red_tea.png'
        yellow_tea_file = Path(__file__).resolve().parent / 'data/misc/yellow_tea.png'
        img_red = imread(red_tea_file)
        img_yellow = imread(yellow_tea_file)
        hist_red = calc_hue_hist(img_red[81:124, 56:132])
        height, width = img.shape[:2]
        hist_target = calc_hue_hist(img[81:124, 56:132])
        score_red = cv2.compareHist(hist_red, hist_target, cv2.HISTCMP_CORREL)
        if score_red > 0.5:
            return ID_RED_TEA
        hist_yellow = calc_hue_hist(img_yellow[81:124, 56:132])
        score_yellow = cv2.compareHist(hist_yellow, hist_target, cv2.HISTCMP_CORREL)
        if score_yellow > 0.5:
            return ID_YELLOW_TEA

        return ID_GREEN_TEA

    def classify_item(self, img, currnet_dropPriority):
        """)
        imgとの距離を比較して近いアイテムを求める
        id を返すように変更
        """
        hash_item = self.hash_item  # 画像の距離
        if logger.isEnabledFor(logging.DEBUG):
            hex = ""
            for h in hash_item[0]:
                hex = hex + "{:02x}".format(h)
            logger.debug("phash: %s", hex)

        def compare_distance(hash_item, background=True):
            ids = {}
            # 既存のアイテムとの距離を比較
            for i in dist_item.keys():
                itemid = dist_item[i]
                item_bg = item_background[itemid]
                d = hasher.compare(hash_item, hex2hash(i))
                if (d <= 20 and dist_item[i] == 1) or (d <= 30 and self.background == "zero"):
                    # QPの背景が誤認識することがあるので背景チェックを回避
                    ids[dist_item[i]] = d
                elif background:
                    if d <= 20 and item_bg == self.background:
                        # ポイントと種の距離が8という例有り(IMG_0274)→16に
                        # バーガーと脂の距離が10という例有り(IMG_2354)→14に
                        ids[dist_item[i]] = d
                else:
                    if d <= 20:
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
                elif id == ID_YELLOW_TEA or id == ID_GREEN_TEA or id == ID_RED_TEA:
                    id = self.classify_tea(img)
                    # logger.info("黄茶葉")

                    # red_tea_file = Path(__file__).resolve().parent / 'data/misc/red_tea.png'
                    # img1 = imread(red_tea_file)
                    # hist1 = calc_red_hist(img1)
                    # height, width = img.shape[:2]
                    # hist2 = calc_red_hist(img[80:height-63, 60:width-72])
                    # score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
                    # logger.info(score)
                    # if score > 0.5:
                    #     return ID_RED_TEA

                return id

            return ""
        id = compare_distance(hash_item, background=True)
        if id == "":
            id = compare_distance(hash_item, background=False)

        return id

    def classify_ce_sub(self, img, hasher_prog, dist_dic, threshold):
        """
        imgとの距離を比較して近いアイテムを求める
        """
        hash_item = hasher_prog(img)  # 画像の距離
        itemfiles = {}
        if logger.isEnabledFor(logging.DEBUG):
            hex = ""
            for h in hash_item[0]:
                hex = hex + "{:02x}".format(h)
        # 既存のアイテムとの距離を比較
        for i in dist_dic.keys():
            d = hasher.compare(hash_item, hex2hash(i))
            if d <= threshold:
                itemfiles[dist_dic[i]] = d
        if len(itemfiles) > 0:
            itemfiles = sorted(itemfiles.items(), key=lambda x: x[1])
            logger.debug("itemfiles: %s", itemfiles)
            item = next(iter(itemfiles))

            return item[0]

        return ""

    def classify_ce(self, img):
        itemid = self.classify_ce_sub(img, compute_hash_ce, dist_ce, 20)
        if itemid == "":
            logger.debug("use narrow image")
            itemid = self.classify_ce_sub(
                        img, compute_hash_ce_narrow, dist_ce_narrow, 20
                        )
        return itemid

    def classify_point(self, img):
        """
        imgとの距離を比較して近いアイテムを求める
        """
        hash_item = compute_hash(img)  # 画像の距離
        itemfiles = {}
        if logger.isEnabledFor(logging.DEBUG):
            hex = ""
            for h in hash_item[0]:
                hex = hex + "{:02x}".format(h)
            logger.debug("phash: %s", hex)
        # 既存のアイテムとの距離を比較
        for i in dist_point.keys():
            itemid = dist_point[i]
            item_bg = item_background[itemid]
            d = hasher.compare(hash_item, hex2hash(i))
            if d <= 20 and item_bg == self.background:
                itemfiles[itemid] = d
        if len(itemfiles) > 0:
            itemfiles = sorted(itemfiles.items(), key=lambda x: x[1])
            item = next(iter(itemfiles))

            return item[0]

        return ""

    def classify_point_and_item(self, img, currnet_dropPriority):
        """
        imgとの距離を比較して近いアイテムを求める
        """
        hash_item = compute_hash(img)  # 画像の距離
        itemfiles = {}
        if logger.isEnabledFor(logging.DEBUG):
            hex = ""
            for h in hash_item[0]:
                hex = hex + "{:02x}".format(h)
            logger.debug("phash: %s", hex)
        # 既存のアイテムとの距離を比較
        dist_item_and_point = dict(**dist_point, **dist_item) 
        for i in dist_item_and_point.keys():
            itemid = dist_item_and_point[i]
            item_bg = item_background[itemid]
            d = hasher.compare(hash_item, hex2hash(i))
            if d <= 30 and itemid == 1 and self.background == "zero":
                itemfiles[itemid] = d
            elif d <= 20 and item_bg == self.background:
                if item_dropPriority[itemid] <= currnet_dropPriority:  # fix #380
                    itemfiles[itemid] = d
        if len(itemfiles) > 0:
            itemfiles = sorted(itemfiles.items(), key=lambda x: x[1])
            item = next(iter(itemfiles))

            if item[0] == ID_YELLOW_TEA or item[0] == ID_GREEN_TEA or item[0] == ID_RED_TEA:
                return self.classify_tea(img)

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
                cv2.imwrite(itemfile.as_posix(), img)
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
                if category == "Craft Essence":
                    hash_narrow = compute_hash_ce_narrow(img)
                    hash_hex_narrow = ""
                    for h in hash_narrow[0]:
                        hash_narrow = hash_narrow + "{:02x}".format(h)
                    dist_ce_narrow[hash_hex_narrow] = id
                item_name[id] = itemfile.stem
                item_background[id] = classify_background(img)
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

    def classify_card(self, img, currnet_dropPriority):
        """
        アイテム判別器
        """
        if self.category == "Point":
            id = self.classify_point(img)
            if id == "":
                id = self.make_new_file(img, Point_dir, dist_point,
                                        PRIORITY_POINT, self.category)
            return id
        elif self.category == "Quest Reward":
            return 5
        elif self.category == "Craft Essence":
            id = self.classify_ce(img)
            if id == "":
                id = self.make_new_file(img, CE_dir, dist_ce,
                                        PRIORITY_CE, self.category)
            return id
        elif self.category == "Exp. UP":
            return self.classify_exp(img)
        elif self.category == "Item":
            id = self.classify_item(img, currnet_dropPriority)
            if id == "":
                id = self.make_new_file(img, Item_dir, dist_item,
                                        PRIORITY_ITEM, self.category)
        else:
            # ここで category が判別できないのは三行目かつ
            # スクロール位置の関係で下部表示が消えている場合
            id = self.classify_point_and_item(img, currnet_dropPriority)
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
        img = img_rgb[int((5+9)/135*self.height):int((30+2)/135*self.height),
                      int(5/135*self.width):int((30+6)/135*self.width)]
        return hasher.compute(img)

    def compute_gem_hash(self, img_rgb):
        """
        スキル石クラス判別器
        中央のクラスマークぎりぎりのハッシュを取る
        記述した比率はiPhone6S画像の実測値
        """
        height, width = img_rgb.shape[:2]

        img = img_rgb[int((145-16-60*0.8)/2/145*height)+2:
                      int((145-16+60*0.8)/2/145*height)+2,
                      int((132-52*0.8)/2/132*width):
                      int((132+52*0.8)/2/132*width)]

        return hasher.compute(img)


def classify_background(img_rgb):
    """
    背景判別
    """
    _, width = img_rgb.shape[:2]
    img = img_rgb[30:119, width - 25:width - 7]
    target_hist = img_hist(img)
    bg_score = []
    score_z = calc_hist_score(target_hist, hist_zero)
    bg_score.append({"background": "zero", "dist": score_z})
    score_g = calc_hist_score(target_hist, hist_gold)
    bg_score.append({"background": "gold", "dist": score_g})
    score_s = calc_hist_score(target_hist, hist_silver)
    bg_score.append({"background": "silver", "dist": score_s})
    score_b = calc_hist_score(target_hist, hist_bronze)
    bg_score.append({"background": "bronze", "dist": score_b})

    bg_score = sorted(bg_score, key=lambda x: x['dist'])
    # logger.debug("background dist: %s", bg_score)
    return (bg_score[0]["background"])


def compute_hash(img_rgb):
    """
    判別器
    この判別器は下部のドロップ数を除いた部分を比較するもの
    記述した比率はiPhone6S画像の実測値
    """
    height, width = img_rgb.shape[:2]
    img = img_rgb[int(23/135*height):
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


def compute_hash_ce_narrow(img_rgb):
    """
    CE Identifier for scrolled down screenshot
    """
    height, width = img_rgb.shape[:2]
    img = img_rgb[int(30/206*height):int(155/206*height),
                  int(5/188*width):int(183/188*width)]
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
        if category == "Item" or category == "Point":
            item_background[id] = classify_background(img)
        if category == "Craft Essence":
            hash_narrow = compute_hash_ce_narrow(img)
            hash_hex_narrow = ""
            for h in hash_narrow[0]:
                hash_hex_narrow = hash_hex_narrow + "{:02x}".format(h)
            dist_ce_narrow[hash_hex_narrow] = id


def calc_hist_score(hist1, hist2):
    scores = []
    for channel1, channel2 in zip(hist1, hist2):
        score = cv2.compareHist(channel1, channel2, cv2.HISTCMP_BHATTACHARYYA)
        scores.append(score)
    return np.mean(scores)


def img_hist(src_img):
    img = cv2.cvtColor(src_img, cv2.COLOR_BGR2HSV)
    hist1 = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([img], [1], None, [256], [0, 256])
    hist3 = cv2.calcHist([img], [2], None, [256], [0, 256])

    return hist1, hist2, hist3


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


def out_name(args, id):
    if args.lang == "eng":
        if id in item_name_eng.keys():
            return item_name_eng[id]
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
        logger.exception(e)
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
    calc_dist_local()
    if train_item.exists() is False:
        logger.critical("item.xml is not found")
        logger.critical("Try to run 'python makeitem.py'")
        sys.exit(1)
    if train_chest.exists() is False:
        logger.critical("chest.xml is not found")
        logger.critical("Try to run 'python makechest.py'")
        sys.exit(1)
    if train_dcnt.exists() is False:
        logger.critical("dcnt.xml is not found")
        logger.critical("Try to run 'python makedcnt.py'")
        sys.exit(1)
    if train_card.exists() is False:
        logger.critical("card.xml is not found")
        logger.critical("Try to run 'python makecard.py'")
        sys.exit(1)
    svm = cv2.ml.SVM_load(str(train_item))
    svm_chest = cv2.ml.SVM_load(str(train_chest))
    svm_dcnt = cv2.ml.SVM_load(str(train_dcnt))
    svm_card = cv2.ml.SVM_load(str(train_card))

    fileoutput = []  # 出力
    output = {}
    prev_pages = 0
    prev_pagenum = 0
    prev_total_qp = QP_UNKNOWN
    prev_itemlist = []
    prev_datetime = datetime.datetime(year=2015, month=7, day=30, hour=0)
    prev_qp_gained = 0
    prev_chestnum = 0
    all_list = []

    for filename in filenames:
        exLogger = CustomAdapter(logger, {"target": filename})

        logger.debug("filename: %s", filename)
        f = Path(filename)

        if f.exists() is False:
            output = {'filename': str(filename) + ': not found'}
            all_list.append([])
        elif f.is_dir():  # for ZIP file from MacOS
            continue
        elif f.suffix.upper() not in ['.PNG', '.JPG', '.JPEG']:
            output = {'filename': str(filename) + ': Not Supported'}
            all_list.append([])
        else:
            img_rgb = imread(filename)
            fileextention = Path(filename).suffix

            try:
                sc = ScreenShot(args, img_rgb,
                                svm, svm_chest, svm_dcnt, svm_card,
                                fileextention, exLogger)
                if sc.itemlist[0]["id"] != ID_REWARD_QP and sc.pagenum == 1:
                    logger.warning(
                                   "Page count recognition is failing: %s",
                                   filename
                                   )
                # ドロップ内容が同じで下記のとき、重複除外
                # QPカンストじゃない時、QPが前と一緒
                # QPカンストの時、Exif内のファイル作成時間が15秒未満
                pilimg = Image.open(filename)
                dt = get_exif(pilimg)
                if dt == "NON" or prev_datetime == "NON":
                    td = datetime.timedelta(days=1)
                else:
                    td = dt - prev_datetime
                if sc.Bunyan:
                    if sc.pages == 1:
                        pass
                    elif sc.lines % 3 == 1:
                        sc.itemlist = sc.itemlist[-1:]
                    else:
                        sc.itemlist = sc.itemlist[7-(sc.lines+1) % 3*7:]
                else:
                    if sc.pages - sc.pagenum == 0:
                        sc.itemlist = sc.itemlist[14-(sc.lines+2) % 3*7:]
                if prev_itemlist == sc.itemlist:
                    if (sc.total_qp != -1 and sc.total_qp != 2000000000
                        and sc.total_qp == prev_total_qp) \
                        or ((sc.total_qp == -1 or sc.total_qp == 2000000000)
                            and td.total_seconds() < args.timeout):
                        logger.debug("args.timeout: %s", args.timeout)
                        logger.debug("filename: %s", filename)
                        logger.debug("prev_itemlist: %s", prev_itemlist)
                        logger.debug("sc.itemlist: %s", sc.itemlist)
                        logger.debug("sc.total_qp: %s", sc.total_qp)
                        logger.debug("prev_total_qp: %s", prev_total_qp)
                        logger.debug("datetime: %s", dt)
                        logger.debug("prev_datetime: %s", prev_datetime)
                        logger.debug("td.total_second: %s", td.total_seconds())
                        fileoutput.append(
                            {'filename': str(filename) + ': duplicate'})
                        all_list.append([])
                        continue

                # 2頁目以前のスクショが無い場合に migging と出力
                # 1. 前頁が最終頁じゃない&前頁の続き頁数じゃない
                # または前頁が最終頁なのに1頁じゃない
                # 2. 前頁の続き頁なのに獲得QPが違う
                if (
                    prev_pages - prev_pagenum > 0
                    and sc.pagenum - prev_pagenum != 1) \
                    or (prev_pages - prev_pagenum == 0
                        and sc.pagenum != 1) \
                    or sc.pagenum != 1 \
                        and sc.pagenum - prev_pagenum == 1 \
                        and (
                                prev_qp_gained != sc.qp_gained
                            ):
                    logger.debug("prev_pages: %s", prev_pages)
                    logger.debug("prev_pagenum: %s", prev_pagenum)
                    logger.debug("sc.pagenum: %s", sc.pagenum)
                    logger.debug("prev_qp_gained: %s", prev_qp_gained)
                    logger.debug("sc.qp_gained: %s", sc.qp_gained)
                    logger.debug("prev_chestnum: %s", prev_chestnum)
                    logger.debug("sc.chestnum: %s", sc.chestnum)
                    fileoutput.append({'filename': 'missing'})
                    all_list.append([])

                all_list.append(sc.itemlist)

                prev_pages = sc.pages
                prev_pagenum = sc.pagenum
                prev_total_qp = sc.total_qp
                prev_itemlist = sc.itemlist
                prev_datetime = dt
                prev_qp_gained = sc.qp_gained
                prev_chestnum = sc.chestnum

                sumdrop = len([d for d in sc.itemlist
                               if d["id"] != ID_REWARD_QP])
                if args.lang == "jpn":
                    drop_count = "ドロ数"
                    item_count = "アイテム数"
                    gained_qp = "獲得QP合計"
                else:
                    drop_count = "item_count"
                    item_count = "item_count"
                    gained_qp = "gained_qp"
                output = {'filename': str(filename), drop_count: sc.chestnum, item_count: sumdrop, gained_qp: sc.qp_gained}

            except Exception as e:
                logger.error(filename)
                logger.error(e, exc_info=True)
                output = ({'filename': str(filename) + ': not valid'})
                all_list.append([])
        fileoutput.append(output)
    return fileoutput, all_list


def sort_files(files, ordering):
    if ordering == Ordering.NOTSPECIFIED:
        return files
    elif ordering == Ordering.FILENAME:
        return sorted(files)
    elif ordering == Ordering.TIMESTAMP:
        return sorted(files, key=lambda f: Path(f).stat().st_ctime)
    raise ValueError(f'Unsupported ordering: {ordering}')


def change_value(args, line):
    if args.lang == 'jpn':
        line = re.sub('000000$', "百万", str(line))
        line = re.sub('0000$', "万", str(line))
        line = re.sub('000$', "千", str(line))
    else:
        line = re.sub('000000$', "M", str(line))
        line = re.sub('000$', "K", str(line))
    return line


def make_quest_output(quest):
    output = ""
    ordeal_call_quest_list = [94086601, 94086602, 94089601, 94089602, 94090701, 94090702, 94093201, 94093202]
    if quest != "":
        quest_list = [q["name"] for q in freequest
                      if q["place"] == quest["place"]]
        if math.floor(quest["id"]/100)*100 == ID_NORTH_AMERICA:
            output = quest["place"] + " " + quest["name"]
        elif math.floor(quest["id"]/100)*100 == ID_SYURENJYO:
            output = quest["chapter"] + " " + quest["place"]
        elif math.floor(quest["id"]/100)*100 == ID_SYURENJYO_TMP:
            output = quest["chapter"] + " " + quest["place"]
        elif (math.floor(quest["id"]/100000)*100000 == ID_EVNET and quest["id"] not in ordeal_call_quest_list) or quest["id"] == ID_WEST_AMERICA_AREA:
            output = quest["shortname"]
        else:
            # クエストが0番目のときは場所を出力、それ以外はクエスト名を出力
            if quest_list.index(quest["name"]) == 0:
                output = quest["chapter"] + " " + quest["place"]
            else:
                output = quest["chapter"] + " " + quest["name"]
    return output


UNKNOWN = -1
OTHER = 0
NOVICE = 1
INTERMEDIATE = 2
ADVANCED = 3
EXPERT = 4
MASTER = 5
ORDEAL_CALL = 6


def tv_quest_type(item_list):
    quest_type = UNKNOWN

    for item in item_list:
        if item["id"] == ID_REWARD_QP:
            if quest_type != UNKNOWN:
                quest_type = OTHER
                break

            if item["dropnum"] == 1400:
                quest_type = NOVICE
            elif item["dropnum"] == 2900:
                quest_type = INTERMEDIATE
            elif item["dropnum"] == 4400:
                quest_type = ADVANCED
            elif item["dropnum"] == 6400:
                quest_type = EXPERT
            elif item["dropnum"] == 7400:
                quest_type = MASTER
            elif item["dropnum"] == 270720:
                quest_type = ORDEAL_CALL
            else:
                quest_type = OTHER
                break
    return quest_type


def deside_tresure_valut_quest(item_list):
    quest_type = tv_quest_type(item_list)
    if quest_type in [UNKNOWN, OTHER]:
        quest_candidate = ""
        return quest_candidate

    item_set = set()
    for item in item_list:
        if item["id"] == ID_REWARD_QP:
            continue
        elif item["id"] != ID_QP:
            quest_candidate = ""
            break
        else:
            item_set.add(item["dropnum"])

    if quest_type == NOVICE and item_set == {10000, 15000, 30000, 45000}:
        quest_candidate = {
                           "id": 94061636,
                           "name": "宝物庫の扉を開け 初級",
                           "place": "",
                           "chapter": "",
                           "qp": 1400,
                           "shortname": "宝物庫 初級",
                           }
    elif quest_type == INTERMEDIATE and item_set == {10000, 15000, 30000, 45000, 90000, 135000}:
        quest_candidate = {
                           "id": 94061637,
                           "name": "宝物庫の扉を開け 中級",
                           "place": "",
                           "chapter": "",
                           "qp": 2900,
                           "shortname": "宝物庫 中級",
                           }
    elif quest_type == ADVANCED and item_set == {30000, 45000, 90000, 135000, 270000, 405000}:
        quest_candidate = {
                           "id": 94061638,
                           "name": "宝物庫の扉を開け 上級",
                           "place": "",
                           "chapter": "",
                           "qp": 4400,
                           "shortname": "宝物庫 上級",
                           }
    elif quest_type == EXPERT and item_set == {90000, 135000, 270000, 405000}:
        quest_candidate = {
                           "id": 94061639,
                           "name": "宝物庫の扉を開け 超級",
                           "place": "",
                           "chapter": "",
                           "qp": 7400,
                           "shortname": "宝物庫 超級",
                           }
    elif quest_type == MASTER and item_set == {270000, 405000, 1500000}:
        quest_candidate = {
                           "id": 94061640,
                           "name": "宝物庫の扉を開け 極級",
                           "place": "",
                           "chapter": "",
                           "qp": 7400,
                           "shortname": "宝物庫 極級",
                           }
    elif quest_type == ORDEAL_CALL and item_set == {270000, 405000, 1500000}:
        quest_candidate = {
                           "id": ID_WEST_AMERICA_AREA,
                           "name": "荒野の歓楽",
                           "place": "アメリカ西部エリア",
                           "chapter": "オーディール・コール",
                           "qp": 270720,
                           "shortname": "オーディール・コール アメリカ西部エリア",
                           }
    else:
        quest_candidate = ""

    return quest_candidate


def deside_quest(item_list):
    quest_name = deside_tresure_valut_quest(item_list)
    if quest_name != "":
        return quest_name

    item_set = set()
    for item in item_list:
        if item["id"] == ID_REWARD_QP:
            item_set.add("QP(+" + str(item["dropnum"]) + ")")
        elif item["id"] == ID_FP:
            continue
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


def quest_name_recognition(item_list):
    """アイテムが全て埋まっていない場合のクエスト名の判別

    Args:
        item_list ([type]): [description]

    Returns:
        [type]: [description]
    """
    reward_qp = 0
    item_set = set()
    for item in item_list:
        if item["id"] == ID_REWARD_QP:
            item_set.add("QP(+" + str(item["dropnum"]) + ")")
            reward_qp = item["dropnum"]
        elif item["id"] == ID_FP:
            continue
        elif item["id"] == 1 \
            or item["category"] == "Craft Essence" \
            or (9700 <= math.floor(item["id"]/1000) <= 9707
                and str(item["id"])[4] not in ["4", "5"]):
            continue
        else:
            item_set.add(item["name"])
    if reward_qp == 0:
        return "", []
    quest_candidate = []
    # 報酬QPが同じクエストの一覧を作る
    for quest in reversed(freequest):
        if quest["qp"] == reward_qp:
            quest_candidate.append(quest)
    # クエスト一覧に含まれるかチェック
    # 含まれるクエストが一つだったら出力
    quest_candidate2 = []
    missing_items = []
    for quest in quest_candidate:
        dropset = {i["name"] for i in quest["drop"]
                   if i["type"] != "Craft Essence"}
        dropset.add("QP(+" + str(quest["qp"]) + ")")
        diff = item_set - dropset
        if len(diff) == 0:
            tmp_items = []
            diff2 = dropset - item_set
            quest_candidate2.append(quest)
            for item in quest["drop"]:
                if item["name"] in diff2:
                    item["dropnum"] = 0
                    item["category"] = "Item"
                    tmp_items.append(item)
            missing_items.append(tmp_items)
    if len(quest_candidate2) == 1:
        return quest_candidate2[0], missing_items[0]
    else:
        return "", []


def make_csv_header(args, item_list):
    """
    CSVのヘッダ情報を作成
    礼装のドロップが無いかつ恒常以外のアイテムが有るとき礼装0をつける
    """
    if args.lang == 'jpn':
        drop_count = 'ドロ数'
        item_count = 'アイテム数'
        gained_qp = '獲得QP合計'
        ce_str = '礼装'
    else:
        drop_count = 'drop_count'
        item_count = 'item_count'
        gained_qp = 'gained_qp'
        ce_str = 'CE'
    sum_files = 0
    for item in item_list:
        sum_files += len(item)
    if sum_files == 0:
        return ['filename', drop_count, item_count, gained_qp], False, ""
    # リストを一次元に
    flat_list = list(itertools.chain.from_iterable(item_list))
    # 余計な要素を除く
    short_list = [{"id": a["id"], "name": a["name"], "category": a["category"],
                   "dropPriority": a["dropPriority"], "dropnum": a["dropnum"]}
                  for a in flat_list]
    # 概念礼装のカテゴリのアイテムが無くかつイベントアイテム(>ID_EXM_MAX)がある
    if args.lang == 'jpn':
        no_ce_exp_list = [
                          k for k in flat_list
                          if not k["name"].startswith("概念礼装EXPカード：")
                          ]
    else:
        no_ce_exp_list = [
                          k for k in flat_list
                          if not k["name"].startswith("CE EXP Card:")
                          ]
    ce0_flag = ("Craft Essence"
                not in [
                        d.get('category') for d in no_ce_exp_list
                       ]
                ) and (
                       max([d.get("id") for d in flat_list]) > ID_EXP_MAX
                )
    if ce0_flag:
        short_list.append({"id": 99999990, "name": ce_str,
                           "category": "Craft Essence",
                           "dropPriority": 9005, "dropnum": 0})
    # 重複する要素を除く
    unique_list = list(map(json.loads, set(map(json.dumps, short_list))))

    # クエスト名判定
    quest = deside_quest(unique_list)
    if quest == "":
        quest, items2 = quest_name_recognition(unique_list)
        unique_list.extend(items2)
    quest_output = make_quest_output(quest)

    # ソート
    new_list = sorted(sorted(sorted(unique_list, key=itemgetter('dropnum')),
                             key=itemgetter('id'), reverse=True),
                      key=itemgetter('dropPriority'), reverse=True)
    header = []
    for nlist in new_list:
        if nlist['category'] in ['Quest Reward', 'Point'] \
           or nlist["name"] == "QP" or nlist["name"] == "フレンドポイント":
            tmp = out_name(args, nlist['id']) \
                  + "(+" + change_value(args, nlist["dropnum"]) + ")"
        elif nlist["dropnum"] > 1:
            tmp = out_name(args, nlist['id']) \
                  + "(x" + change_value(args, nlist["dropnum"]) + ")"
        elif nlist["name"] == ce_str:
            tmp = ce_str
        else:
            tmp = out_name(args, nlist['id'])
        header.append(tmp)
    return ['filename', drop_count, item_count, gained_qp] + header, ce0_flag, quest_output


def make_csv_data(args, sc_list, ce0_flag):
    if sc_list == []:
        return [{}], [{}]
    csv_data = []
    allitem = []
    for sc in sc_list:
        tmp = []
        for item in sc:
            if item['category'] in ['Quest Reward', 'Point'] \
               or item["name"] == "QP" or item["name"] == "フレンドポイント":
                tmp.append(out_name(args, item['id'])
                           + "(+" + change_value(args, item["dropnum"]) + ")")
            elif item["dropnum"] > 1:
                tmp.append(out_name(args, item['id'])
                           + "(x" + change_value(args, item["dropnum"]) + ")")
            else:
                tmp.append(out_name(args, item['id']))
        allitem = allitem + tmp
        csv_data.append(dict(Counter(tmp)))
    csv_sum = dict(Counter(allitem))
    if ce0_flag:
        if args.lang == 'jpn':
            ce_str = '礼装'
        else:
            ce_str = 'CE'
        csv_sum.update({ce_str: 0})
    return csv_sum, csv_data


def list_to_dict(lst):
    result = {}
    for item in lst:
        result[item] = 0
    return result


if __name__ == '__main__':
    # オプションの解析
    parser = argparse.ArgumentParser(
                        description='Image Parse for FGO Battle Results'
                        )
    # 3. parser.add_argumentで受け取る引数を追加していく
    parser.add_argument('filenames',
                        help='Input File(s)', nargs='*')    # 必須の引数を追加
    parser.add_argument('--lang', default=DEFAULT_ITEM_LANG,
                        choices=('jpn', 'eng'),
                        help='Language to be used for output: Default '
                             + DEFAULT_ITEM_LANG)
    parser.add_argument('-f', '--folder', help='Specify by folder')
    parser.add_argument('--ordering',
                        help='The order in which files are processed ',
                        type=Ordering,
                        choices=list(Ordering), default=Ordering.NOTSPECIFIED)
    text_timeout = 'Duplicate check interval at QP MAX (sec): Default '
    parser.add_argument('-t', '--timeout', type=int, default=TIMEOUT,
                        help=text_timeout + str(TIMEOUT) + ' sec')
    parser.add_argument('--version', action='version',
                        version=PROGNAME + " " + VERSION)
    parser.add_argument('-l', '--loglevel',
                        choices=('debug', 'info'), default='info')

    args = parser.parse_args()    # 引数を解析
    lformat = '%(name)s <%(filename)s-L%(lineno)s> [%(levelname)s] %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=lformat,
    )
    logger.setLevel(args.loglevel.upper())

    for ndir in [Item_dir, CE_dir, Point_dir]:
        if not ndir.is_dir():
            ndir.mkdir(parents=True)

    if args.folder:
        inputs = [x for x in Path(args.folder).glob(r"**/[!.]*")]
    else:
        inputs = args.filenames

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf_8_sig')

    inputs = sort_files(inputs, args.ordering)
    fileoutput, all_new_list = get_output(inputs, args)
    if len(all_new_list) == 0:
        print("filename,ドロ数\n合計,0\nファイルが見つかりません,\n")
        exit(0)

    # CSVヘッダーをつくる
    csv_header, ce0_flag, questname = make_csv_header(args, all_new_list)
    csv_sum, csv_data = make_csv_data(args, all_new_list, ce0_flag)
    a = list_to_dict(csv_header)

    writer = csv.DictWriter(sys.stdout, fieldnames=csv_header,
                            lineterminator='\n')
    writer.writeheader()
    if args.lang == 'jpn':
        drop_count = 'ドロ数'
        item_count = 'アイテム数'
        gained_qp = '獲得QP合計'
    else:
        drop_count = 'drop_count'
        item_count = 'item_count'
        gained_qp = 'gained_qp'
    if len(all_new_list) > 0:
        if questname == "":
            if args.lang == 'jpn':
                questname = "合計"
            else:
                questname = "SUM"
        a.update({'filename': questname, drop_count: '', item_count: '', gained_qp: ''})
        a.update(csv_sum)
        writer.writerow(a)
    for fo, cd in zip(fileoutput, csv_data):
        fo.update(cd)
        writer.writerow(fo)
    if drop_count in fo.keys():  # issue: #55
        if len(fileoutput) > 1 and str(fo[drop_count]).endswith('+'):
            writer.writerow({'filename': 'missing'})
