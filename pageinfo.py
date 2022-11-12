#!/usr/bin/env python3
#
# MIT License
# Copyright 2020-2022 max747
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the 
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#

import argparse
import csv
import enum
import logging
import math
import os
import sys
from pathlib import Path

import cv2  # type: ignore

logger = logging.getLogger(__name__)
pageinfo_basedir = Path(__file__).parent

NOSCROLL_PAGE_INFO = (1, 1, 0)

SCRB_LIKELY_SCROLLBAR = 1   # スクロールバーと推定
SCRB_TOO_SMALL = 2  # 領域が小さすぎる
SCRB_TOO_THICK = 3  # 領域の横幅が太すぎる
SCRB_TOO_FAR_FROM_CENTER = 4    # 中央から遠すぎる
SCRB_TOO_MANY_VERTICES = 5  # 頂点が多すぎる
SCRB_TOO_FAR_FROM_LEFT_EDGE = 6  # 左端から遠すぎる
SCRB_TOO_CLOSE_TO_LEFT_EDGE = 7  # 左端に近すぎる
SCRB_TOO_THIN = 8   # 領域の横幅が細すぎる

GS_TYPE_1 = 1   # 旧画面
GS_TYPE_2 = 2   # wide screen 対応画面。戦利品ウィンドウの位置が上にシフトした


class QPDetectionMode(enum.Enum):
    JP = 'jp'
    NA = 'na'

    def __str__(self):
        return self.value

    @classmethod
    def values(cls):
        return [str(e) for e in list(cls)]


class PageInfoError(Exception):
    pass


class TooManyAreasDetectedError(PageInfoError):
    pass


class UnsupportedGamescreenTypeError(PageInfoError):
    pass


def detect_side_black_margin(im_gray):
    """
        画像の左右にある黒領域を検出する。
        それぞれの幅サイズを (右幅, 左幅) のタプルで返す。
        余白がなければ (0, 0) を返す。
    """
    height, width = im_gray.shape[:2]
    # 黒とみなす範囲: 0 に近いほど許容範囲が小さい
    black_threshold = 10
    # タップの軌跡などノイズが混入する可能性もあるので 12 % まではイレギュラーを許容する
    black_ratio = 0.88

    for i in range(width):
        black_pixels = sum([pixel < black_threshold for pixel in im_gray[:, i]])
        if black_pixels / height < black_ratio:
            break

    left_margin = i

    for j in range(width):
        black_pixels = sum([pixel < black_threshold for pixel in im_gray[:, width - j - 1]])
        if black_pixels / height < black_ratio:
            break

    right_margin = j

    # 真っ黒画像の場合はマージンなしとする
    if left_margin + right_margin >= width:
        logger.warning("no margins: completely black image")
        return 0, 0

    return left_margin, right_margin


def filter_contour_qp(contour, im):
    """
        "所持 QP" エリアを拾い、それ以外を除外するフィルター
    """
    im_h, im_w = im.shape[:2]
    # 画像全体に対する検出領域の面積比が一定以上であること。
    # 明らかに小さすぎる領域はここで捨てる。
    if cv2.contourArea(contour) * 25 < im_w * im_h:
        return False
    x, y, w, h = cv2.boundingRect(contour)
    # 横長領域なので、高さに対して十分大きい幅になっていること。
    if w < h * 6:
        return False
    # 横幅が画像サイズに対して長すぎず短すぎないこと。
    # 長すぎる場合は画面下部の端末別表示調整用領域を検出している可能性がある。
    if not (w * 1.2 < im_w < w * 2):
        return False
    # 検出位置が端に寄りすぎていないこと。
    # 上
    margin_h = im_h // 10
    if y < margin_h:
        logger.debug('too close to the top edge')
        return False
    # 下
    if y + h > im_h - margin_h:
        logger.debug('too close to the bottom edge')
        return False
    # 左
    margin_w = im_w // 10
    if x < margin_w:
        logger.debug('too close to the left edge')
        return False
    # 右はチェックしない。
    # もともとのレイアウトの関係で右に寄りがちなので、誤判定の可能性が高くなる。
    # また、右に寄りすぎている領域の検出自体が考えにくい。

    logger.debug('qp region: (x, y, width, height) = (%s, %s, %s, %s)', x, y, w, h)
    return True


def detect_qp_region(im, mode=QPDetectionMode.JP.value, debug_draw_image=False, debug_image_name=None):
    """
        "所持 QP" 領域を検出し、その座標を返す。

        戻り値は (左上座標, 右下座標)
        つまり ((topleft_x, topleft_y), (bottomright_x, bottomright_y))
        領域が検出されなかった場合は None を返す。
        複数箇所が検出された場合は TooManyAreasDetectedError が発生する。
    """
    # 縦横2分割して4領域に分け、左下の領域だけ使う。
    # QP の領域を調べたいならそれで十分。
    im_h, im_w = im.shape[:2]
    # まれにスクリーンショット左端に余白が入ることがある。
    # おそらく Android の機種や状況に依存？ この状態で左右を等分すると
    # 中心が左にずれて QP 領域の囲みが見切れてしまい検出に失敗する。
    # これを考慮し、切る位置をやや右にずらす。
    cropped = im[int(im_h/2):im_h, 0:int(im_w/1.93)]
    cr_h, cr_w = cropped.shape[:2]
    logger.debug('cropped image size (for qp): (width, height) = (%s, %s)', cr_w, cr_h)
    im_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    binary_threshold = 50
    _, th1 = cv2.threshold(im_gray, binary_threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(th1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filtered_contours = [c for c in contours if filter_contour_qp(c, im_gray)]
    candidate = None

    for contour in filtered_contours:
        logger.debug('detected areas: %s', cv2.boundingRect(contour))

    if len(filtered_contours) == 1:
        qp_region = filtered_contours[0]
        x, y, w, h = cv2.boundingRect(qp_region)

        wh_rate = w / h

        # 左右の無駄領域を除外するためのマージン。
        #
        # The position of the QP values in the NA version of the screenshot is
        # slightly more to the right than in the JP version. This makes it
        # difficult to apply the same cut position to both types of screenshots.
        if mode == QPDetectionMode.NA.value:
            # The values below are optimized for NA's new game screen layout.
            # Old layout screenshots can also be applied, but may not cut well.
            left_margin = 0.45
            right_margin = 0.02
        else:
            # イベントで所持 QP 枠が狭くなる場合、カットする領域を狭める必要がある。
            if wh_rate < 9:
                left_margin = 0.45
                right_margin = 0.04
            else:
                # 感覚的な値ではあるが 左 42%, 右 4% を除外。
                # 落とし穴として、2019年5月末 ～ 9月の間に所持 QP の出力位置が微妙に変わっている。
                # ここではそのどちらのケースでも対応できるよう枠を広めに取っている。
                # 現仕様に最適化して切り詰めすぎると困ったことになるため注意。
                left_margin = 0.42
                right_margin = 0.04

        topleft = (x + int(w*left_margin), y)
        bottomright = (topleft[0] + w - int(w*left_margin) - int(w*right_margin), y + h)

        if debug_draw_image:
            cv2.rectangle(cropped, topleft, bottomright, (0, 0, 255), 3)

        # 呼び出し側に返すのは分割前の座標でないといけない。
        # よって、事前に切り捨てた左上領域の y 座標をここで補正する。
        # x 座標は分割の影響を受けていないので補正不要。
        above_height = int(im_h/2)
        corrected_topleft = (topleft[0], topleft[1] + above_height)
        corrected_bottomright = (bottomright[0], bottomright[1] + above_height)
        candidate = (corrected_topleft, corrected_bottomright)

    if debug_draw_image:
        cv2.drawContours(cropped, filtered_contours, -1, (0, 255, 0), 3)
        # 以下はどうしてもデバッグ目的で輪郭検出の状況を知りたい場合のみ有効にする。
        # 通常はコメントアウトしておく。
        # cv2.drawContours(cropped, contours, -1, (0, 255, 255), 3)
        logger.debug('writing debug image: %s', debug_image_name)
        cv2.imwrite(debug_image_name, cropped)

    if len(filtered_contours) > 1:
        n = len(filtered_contours)
        raise TooManyAreasDetectedError(f'{n} actual qp regions detected')

    return candidate


def guess_pages(actual_height, entire_height, cap_height):
    """
        スクロールバー領域の高さからドロップ枠が何ページあるか推定する
    """
    inner_height = actual_height - cap_height * 2
    ratio = inner_height / entire_height
    logger.debug('guess_pages> inner_height: %s, entire_height: %s, ratio: %s', actual_height, entire_height, ratio)
    if ratio > 0.8:
        return 1
    elif ratio > 0.46:
        return 2
    elif ratio > 0.316:  # 9列のとき 0.361-0.364 くらい
        return 3
    elif ratio > 0.24:
        return 4
    elif ratio > 0.193:
        return 5
    # 高々 6 ページ (ドロップ枠総数 <= 125) と仮定。
    return 6


def guess_pagenum(actual_y, entire_y, actual_height, entire_height, cap_height):
    """
        スクロールバーの y 座標の位置および高さからドロップ画像のページ数を推定する
    """

    # スクロールバーと上端との空き領域の縦幅 delta と
    # スクロール可能領域の縦幅 entire_height との比率で位置を推定する。
    delta = (actual_y + cap_height) - entire_y
    inner_height = actual_height - cap_height * 2
    height_ratio = inner_height / entire_height
    ratio = delta / entire_height
    logger.debug(
        'guess_pagenum> space above scrollbar: %s, actual_y: %s, entire_y: %s, inner_height: %s, entire_height: %s, ratio: %s, height_ratio: %s, r/h = %s',
        delta, actual_y, entire_y, inner_height, entire_height, ratio, height_ratio, ratio / height_ratio,
    )
    rh = ratio / height_ratio
    # 単純な四捨五入だと最下段の処理でうまくいかない。
    # 最下段は1行だけのこともあり、その場合はスクロールの量が少なくなるため。
    # 切り上げのラインを低く設定するため独自に切り上げ処理を記述する。
    # rh の小数部が 0.2 以上なら切り上げとする。
    # ただし rh の整数部が大きい場合 (=ページ数が多い場合) は 0.2 でも
    # 誤判定する可能性が出てくるため、切り上げラインを 0.1 にする。
    if int(rh) < 3:
        return int((rh * 5 + 4) / 5) + 1
    else:
        return int((rh * 10 + 9) / 10) + 1


def guess_lines(actual_height, entire_height, cap_height):
    """
        スクロールバー領域の高さからドロップ枠が何行あるか推定する
        スクロールバーを用いる関係上、原理的に 2 行以下は推定不可
    """
    ratio = (actual_height - cap_height * 2) / entire_height
    logger.debug('guess_lines> scrollbar ratio: %s', ratio)

    if ratio > 0.89:
        return 3
    elif ratio > 0.65:  # 実測値 0.688
        return 4
    elif ratio > 0.53:  # 実測値 0.556
        return 5    # -34
    elif ratio > 0.44:  # 実測値 0.466
        return 6    # -41
    elif ratio > 0.39:  # 実測値 0.403-0.405
        return 7    # -48
    elif ratio > 0.34:  # 実測値 0.355-0.358
        return 8    # -55
    elif ratio > 0.31:  # 実測値 0.3192-0.3224
        return 9    # -62
    elif ratio > 0.284:  # 実測値 0.2909-0.2926
        return 10   # -69
    elif ratio > 0.261:  # 実測値 0.2669-0.2698
        return 11   # -76
    elif ratio > 0.244:  # 実測値 0.2471-0.2514
        return 12   # -83
    elif ratio > 0.225:  # 実測値 0.2302-0.2343
        return 13   # -90
    elif ratio > 0.215:  # 実測値 0.2164-0.2203
        return 14   # -97
    elif ratio > 0.191:  # 実測値 0.1932-0.2142
        return 15   # -104
    else:
        # 15 行以上は考慮しない
        return 16


def _filter_contour_scrollbar(contour, im_height, im_width):
    """
        スクロールバー領域を拾い、それ以外を除外するフィルター

        適合する場合は contour オブジェクトを、適合しない場合は None を返す。
    """
    # 画像全体に対する検出領域の面積比が一定以上であること。
    # 明らかに小さすぎる領域はここで捨てる。
    if cv2.contourArea(contour) * 120 < im_height * im_width:
        return SCRB_TOO_SMALL

    x, y, w, h = cv2.boundingRect(contour)
    logger.debug('scrollbar candidate: (x, y, width, height) = (%s, %s, %s, %s)', x, y, w, h)
    # 縦長領域なので、幅に対して十分大きい高さになっていること。
    if h < w * 3:
        logger.debug("NG: not enough to high: height %s < width %s * 3", h, w)
        return SCRB_TOO_THICK

    # 画面全体 (上下カット済み) の高さに対してスクロールバーの幅が一定の比率に収まること。
    width_ratio = w / im_height
    # NOTE: NA 版の細いスクロールバーを許容しようとすると閾値 0.035 では厳しい。
    # NA/JP の情報をこの位置まで引き継ぐことができれば閾値をコントロールできるが...
    if width_ratio < 0.020:
        logger.debug("NG: too thin: width = %s, im_height = %s, ratio = %s", w, im_height, width_ratio)
        return SCRB_TOO_THIN

    if width_ratio > 0.045:
        logger.debug("NG: too thick: width = %s, im_height = %s, ratio = %s", w, im_height, width_ratio)
        return SCRB_TOO_THICK

    # 検出領域の x 座標が画面端よりも中央線に近いこと。
    # wide screen の場合は逆に端に寄ってしまう。wide screen かどうかを先に調べて、その場合は離れているかどうかをテストする。
    width_range = im_width / 4

    if im_width / im_height > 0.55:
        # wide screen
        # この場合、スクロールバーは左端に寄った形になるはず。
        if x > width_range:
            logger.debug(
                "NG: far from left edge: potition x = %s, width = %s, range = %s",
                x, im_width, width_range,
            )
            return SCRB_TOO_FAR_FROM_LEFT_EDGE

        # 左端のアイテム画像の切れ端をスクロールバーと誤認する問題があるため、左端を検出した場合は落とす。
        if x < width_range / 2:
            logger.debug(
                "NG: too close to left edge: potition x = %s, width = %s, range = %s",
                x, im_width, width_range / 2,
            )
            return SCRB_TOO_CLOSE_TO_LEFT_EDGE

    else:
        # normal screen
        if abs(x - im_width / 2) > width_range:
            logger.debug(
                "NG: far from center line: potition x = %s, width = %s, center = %s, range = %s",
                x, im_width, im_width / 2, width_range,
            )
            return SCRB_TOO_FAR_FROM_CENTER

    # 頂点の数が多すぎないこと。
    if len(contour) > 150:
        # 背景の影響でジャギーなラインになってしまうケースがあるため、シンプルな形状に近似する
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) > 50:
            logger.debug("NG: too many vertices: %s", len(approx))
            return SCRB_TOO_MANY_VERTICES

    logger.debug('found')
    return SCRB_LIKELY_SCROLLBAR


def _detect_scrollbar_region(im, binary_threshold):
    _, th1 = cv2.threshold(im, binary_threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(th1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    im_height, im_width = im.shape[:2]

    scrollbar = []
    not_scrollbar = []

    for c in contours:
        result = _filter_contour_scrollbar(c, im_height, im_width)
        if result == SCRB_LIKELY_SCROLLBAR:
            scrollbar.append(c)
        elif result == SCRB_TOO_THICK:
            not_scrollbar.append(c)
    return scrollbar, not_scrollbar


def get_gamescreen_type(im_cropped, button):
    try:
        res = cv2.matchTemplate(im_cropped, button, cv2.TM_CCOEFF_NORMED)
    except cv2.error:
        # 次へボタンが検出できない場合は新画面であると仮定する。
        # アプリの用途から考えて、旧画面の画像が投入される可能性はきわめてまれ。
        return GS_TYPE_2

    _, _, _, coord = cv2.minMaxLoc(res)
    logger.debug("next button: (top, left) = %s", coord)
    y = coord[1]
    button_bottom_pos = y + button.shape[0]
    im_height = im_cropped.shape[0]
    button_space_height = im_height - button_bottom_pos

    bottom_space_ratio = button_space_height / im_height
    logger.debug("buttom space ratio: %s", bottom_space_ratio)

    # "次へ" ボタンの下の空間が大きければ新画面
    if bottom_space_ratio < 0.05:
        return GS_TYPE_1
    return GS_TYPE_2


def _imwrite_debug(filename, im, suffix):
    base, ext = os.path.splitext(filename)
    name = f"{base}_{suffix}{ext}"
    cv2.imwrite(name, im)


def _try_to_detect_scrollbar(im_gray, im_for_debug=None, debug_image_name="", **kwargs):
    """
        スクロールバーおよびスクロール可能領域の検出

        debug 画像を出力したい場合は im_orig_for_debug に二値化
        される前の元画像 (crop されたもの) を渡すこと。
    """
    # 二値化の閾値を高めにするとスクロールバー本体の領域を検出できる。
    # 低めにするとスクロールバー可能領域を検出できる。
    threshold_for_actual = 65

    actual_scrollbar_contours, not_scrollbar_contours = _detect_scrollbar_region(im_gray, threshold_for_actual)
    if im_for_debug is not None and debug_image_name:
        cv2.drawContours(im_for_debug, not_scrollbar_contours, -1, (0, 255, 64), 2)
        _imwrite_debug(debug_image_name, im_for_debug, "not_scrollbar")

    if len(actual_scrollbar_contours) == 0:
        return None

    if im_for_debug is not None:
        cv2.drawContours(im_for_debug, actual_scrollbar_contours, -1, (0, 255, 0), 3)
        _imwrite_debug(debug_image_name, im_for_debug, "scrollbar")

    if len(actual_scrollbar_contours) > 1:
        n = len(actual_scrollbar_contours)
        raise TooManyAreasDetectedError(f'{n} actual scrollbar areas are detected')

    return actual_scrollbar_contours[0]


def _compute_scrollable_area_position_and_height(im_height, gamescreen_type):
    """
        画像の高さからスクロール可能領域のy座標位置と高さを計算して返す。
    """
    if gamescreen_type == GS_TYPE_1:
        vertical_position_rate = 0.174
        height_rate = 0.557
    elif gamescreen_type == GS_TYPE_2:
        vertical_position_rate = 0.122
        height_rate = 0.56
    else:
        raise UnsupportedGamescreenTypeError("unsupported gamescreen type: %s", gamescreen_type)

    vertical_position = round(im_height * vertical_position_rate)
    scrollable_area_height = round(im_height * height_rate)
    logger.debug("scrollable area y position: %s, height: %s", vertical_position, scrollable_area_height)
    return (vertical_position, scrollable_area_height)


def _compute_scrollbar_cap_height(im_height):
    cap_height = im_height * 0.0122
    logger.debug("cap height: %s (%s)", int(cap_height), cap_height)
    return int(cap_height)


def guess_pageinfo(im, debug_draw_image=False, debug_image_name=None, **kwargs):
    """
        ページ情報を推定する。
        返却値は (現ページ数, 全体ページ数, 全体行数)
        スクロールバーがない場合は全体行数の推定は不可能。その場合は
        NOSCROLL_PAGE_INFO すなわち (1, 1, 0) を返す
    """
    im_h, im_w = im.shape[:2]
    logger.debug('image size: (width, height) = (%s, %s)', im_w, im_h)

    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    left_margin, right_margin = detect_side_black_margin(im_gray)
    logger.debug('side margin: (left, right) = (%s, %s)', left_margin, right_margin)

    # 左右に黒領域がある場合、まずこれを除去する。
    left = left_margin
    right = im_w - right_margin
    net_width = right - left
    logger.debug('net_width = %s', net_width)

    # 縦横比率が規定値を超える場合は上下カットが必要と判断する。
    if im_h / net_width > 0.57:
        cut_size = int(math.ceil(int(im_h - net_width * 0.56) / 2))
        top = cut_size
        bottom = im_h - cut_size
    else:
        cut_size = 0
        top = 0
        bottom = im_h

    logger.debug('top, bottom = (%s, %s), cut_size = %s', top, bottom, cut_size)
    # 縦4分割して4領域に分け、一番右の領域だけ使う。
    # スクロールバーの領域を調べたいならそれで十分。
    cropped = im[top:bottom, int(net_width*3/4):net_width]
    cr_h, cr_w = cropped.shape[:2]
    logger.debug('cropped image size (for scrollbar): (width, height) = (%s, %s)', cr_w, cr_h)
    cropped_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    next_button = cv2.imread(str(pageinfo_basedir / "data" / "pageinfo" / "next.png"))
    next_button_gray = cv2.cvtColor(next_button, cv2.COLOR_BGR2GRAY)
    gamescreen_type = get_gamescreen_type(cropped_gray, next_button_gray)

    if debug_draw_image:
        im_orig_for_debug = cropped
    else:
        im_orig_for_debug = None

    try:
        actual_scrollbar_region = _try_to_detect_scrollbar(cropped_gray, im_orig_for_debug, debug_image_name=debug_image_name, **kwargs)
    finally:
        if debug_draw_image:
            logger.debug('writing debug image: %s', debug_image_name)
            cv2.imwrite(debug_image_name, cropped)

    # スクロールバーが検出できない
    if actual_scrollbar_region is None:
        return NOSCROLL_PAGE_INFO

    _, asr_y, _, asr_h = cv2.boundingRect(actual_scrollbar_region)

    esr_y, esr_h = _compute_scrollable_area_position_and_height(cr_h, gamescreen_type)
    cap_height = _compute_scrollbar_cap_height(bottom - top)
    pages = guess_pages(asr_h, esr_h, cap_height)
    pagenum = guess_pagenum(asr_y, esr_y, asr_h, esr_h, cap_height)
    lines = guess_lines(asr_h, esr_h, cap_height)
    return (pagenum, pages, lines)


def look_into_file_for_page(filename, im, args):
    if args.debug_sc:
        debug_sc_dir = os.path.join(args.debug_out_dir, 'page')
        os.makedirs(debug_sc_dir, exist_ok=True)
        prefix = args.debug_out_file_prefix
        debug_image = os.path.join(debug_sc_dir, prefix + os.path.basename(filename))
        logger.debug('debug image path: %s', debug_image)
    else:
        debug_image = None

    pagenum, pages, lines = guess_pageinfo(im, args.debug_sc, debug_image)
    logger.debug('pagenum: %s, pages: %s, lines: %s', pagenum, pages, lines)
    return (pagenum, pages, lines)


def look_into_file_for_qp(filename, im, args):
    if args.debug_sc:
        debug_sc_dir = os.path.join(args.debug_out_dir, 'qp')
        os.makedirs(debug_sc_dir, exist_ok=True)
        prefix = args.debug_out_file_prefix
        debug_image = os.path.join(debug_sc_dir, prefix + os.path.basename(filename))
        logger.debug('debug image path: %s', debug_image)
    else:
        debug_image = None
    result = detect_qp_region(im, args.mode, args.debug_sc, debug_image)
    if result is None:
        return ('', '') , ('', '')
    return result


def look_into_file(filename, args):
    logger.debug(f'===== {filename}')

    im = cv2.imread(filename)
    if im is None:
        raise FileNotFoundError(f'Cannot read file: {filename}')

    im_h, im_w = im.shape[:2]
    logger.debug('image size: (width, height) = (%s, %s)', im_w, im_h)

    return args.func(filename, im, args)


def main(args):
    csvdata = []

    for filename in args.filename:
        if os.path.isdir(filename):
            for child in os.listdir(filename):
                path = os.path.join(filename, child)
                result = look_into_file(path, args)
                csvdata.append((path, *result))
        else:
            result = look_into_file(filename, args)
            csvdata.append((filename, *result))

    csv_writer = csv.writer(args.output, lineterminator='\n')
    csv_writer.writerows(csvdata)


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    def add_common_arguments(p):
        p.add_argument('filename', nargs='+')
        p.add_argument(
            '-l', '--loglevel',
            choices=('debug', 'info', 'warning'),
            default='info',
            help='set loglevel [default: info]',
        )
        p.add_argument(
            '-ds', '--debug-sc',
            action='store_true',
            help='enable writing sc image for debug',
        )
        p.add_argument(
            '-do', '--debug-out-dir',
            default='debugimages',
            help='output directory for debug images [default: debugimages]',
        )
        p.add_argument(
            '-dp', '--debug-out-file-prefix',
            default='',
            help='filename prefix for debug image [default: "" (no prefix)]'
        )
        p.add_argument(
            '-o', '--output',
            type=argparse.FileType('w'),
            default=sys.stdout,
            help='output file [default: STDOUT]',
        )

    page_parser = subparsers.add_parser('page')
    add_common_arguments(page_parser)
    page_parser.set_defaults(func=look_into_file_for_page)

    qp_parser = subparsers.add_parser('qp')
    add_common_arguments(qp_parser)
    qp_parser.add_argument(
        '-m',
        '--mode',
        choices=QPDetectionMode.values(),
        default=QPDetectionMode.JP.value,
    )
    qp_parser.set_defaults(func=look_into_file_for_qp)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    logger.setLevel(args.loglevel.upper())
    main(args)
