#!/usr/bin/env python3
#
# MIT License
# Copyright 2020 max747
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

import cv2

logger = logging.getLogger(__name__)

NOSCROLL_PAGE_INFO = (1, 1, 0)


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
    return int((rh * 5 + 4) / 5) + 1


def guess_lines(actual_height, entire_height, cap_height):
    """
        スクロールバー領域の高さからドロップ枠が何行あるか推定する
        スクロールバーを用いる関係上、原理的に 2 行以下は推定不可
    """
    ratio = (actual_height - cap_height * 2) / entire_height
    logger.debug('guess_lines> scrollbar ratio: %s', ratio)

    if ratio > 0.90:
        return 3
    elif ratio > 0.65:  # 実測値 0.688
        return 4
    elif ratio > 0.53:   # 実測値 0.556
        return 5    # -34
    elif ratio > 0.44:   # 実測値 0.466
        return 6    # -41
    elif ratio > 0.39:  # 実測値 0.403-0.405
        return 7    # -48
    elif ratio > 0.34:  # 実測値 0.355-0.358
        return 8    # -55
    elif ratio > 0.31:  # 実測値 0.3192-0.3224
        return 9    # -62
    elif ratio > 0.285:  # 実測値 0.2909-0.2926
        return 10   # -69
    elif ratio > 0.262:  # 実測値 0.2669-0.2698
        return 11   # -76
    elif ratio > 0.245:
        return 12   # -83
    elif ratio > 0.228: 
        return 13   # -90
    elif ratio > 0.214:  # 実測値 0.2164-0.2203
        return 14   # -97
    elif ratio > 0.191:  # 実測値 0.1932-0.2132
        return 15   # -104
    else:
        # 15 行以上は考慮しない
        return 16


SCRB_LIKELY_SCROLLBAR = 1  # スクロールバーと推定
SCRB_TOO_SMALL = 2  # 領域が小さすぎる
SCRB_TOO_THICK = 3  # 領域の横幅が太すぎる


def filter_contour_scrollbar(contour, im_height, im_width):
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
        logger.debug("enough to high: height %s < width %s * 3", h, w)
        return SCRB_TOO_THICK

    logger.debug('scrollbar region: (x, y, width, height) = (%s, %s, %s, %s)', x, y, w, h)
    logger.debug('found')
    return SCRB_LIKELY_SCROLLBAR


def filter_contour_scrollable_area(contour, scrollbar_contour, im):
    """
        スクロール可能領域を拾い、それ以外を除外するフィルター

        適合する場合は contour オブジェクトを、適合しない場合は None を返す。
        ただし contour オブジェクトは近似図形に補正されることがある。
    """
    im_w, im_h = im.shape[:2]
    # 画像全体に対する検出領域の面積比が一定以上であること。
    # 明らかに小さすぎる領域はここで捨てる。
    if cv2.contourArea(contour) * 50 < im_w * im_h:
        return None
    x, y, w, h = cv2.boundingRect(contour)
    logger.debug('scrollable area candidate: (x, y, width, height) = (%s, %s, %s, %s)', x, y, w, h)

    sx, sy, sw, sh = cv2.boundingRect(scrollbar_contour)

    # スクロールバーより高さが低いものはダメ
    if h < sh:
        logger.debug('NG: height %s is less than scrollbar %s', h, sh)
        return None

    # 長方形に近似する
    epsilon = 0.05 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    ax, ay, aw, ah = cv2.boundingRect(approx)
    logger.debug('approx rectangle: (x, y, width, height) = (%s, %s, %s, %s)', ax, ay, aw, ah)

    # 近似図形の幅はスクロールバーの幅+5以下
    if aw > sw + 5:
        logger.debug('NG: approx width %s is greater than scrollbar %s + 5', aw, sw)
        return None

    # 近似図形の x 座標はスクロールバーの幅 + 10 に収まること
    if ax < sx - 10:
        logger.debug('NG: approx x %s is more left than scrollbar %s - 10 = %s', ax, sx, sx - 10)
        return None
    if ax > sx + sw + 10:
        logger.debug('NG: approx x %s is more right than scrollbar %s + width %s + 10 = %s', ax, sx, sw, sx + sw + 10)
        return None

    # 近似図形の上端はスクロールバーよりも高い位置
    if ay > sy:
        logger.debug('NG: approx top position %s is under the scrollbar top %s', ay, sy)
        return None
    # 近似図形の下端はスクロールバーよりも低い位置
    if ay + ah < sy + sh:
        logger.debug('NG: approx bottom position %s is over the scrollbar bottom %s', ay + ah, sy + sh)
        return None

    # 近似図形の高さはスクロールバーの幅の13倍以上16倍以下
    if sw * 13 > ah:
        logger.debug('NG: approx height %s is less than scrollbar width %s * 13 = %s', ah, sw, sw * 13)
        return None
    # TODO このアルゴリズムでは NA 版の昔の形式のスクリーンショットはうまく解釈できない。
    # なぜなら、NA 版の昔の形式のスクリーンショットは幅がとても細いため、スクロールバーの
    # 幅の16倍以下という条件をクリアできないから。
    # 一時的にこの制限を外して検証してみるか？ 幅とx座標の位置を事前に検証済みなので
    # 16倍以下の制限はなくてもいいかもしれない。
    # if sw * 16 < ah:
    #    return None

    logger.debug('found')
    return approx


def _detect_scrollbar_region(im, binary_threshold):
    _, th1 = cv2.threshold(im, binary_threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(th1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    im_height, im_width = im.shape[:2]

    scrollbar = []
    not_scrollbar = []

    for c in contours:
        result = filter_contour_scrollbar(c, im_height, im_width)
        if result == SCRB_LIKELY_SCROLLBAR:
            scrollbar.append(c)
        elif result == SCRB_TOO_THICK:
            not_scrollbar.append(c)
    return scrollbar, not_scrollbar


def _detect_scrollable_area(im, binary_threshold, scrollbar_contour):
    _, th1 = cv2.threshold(im, binary_threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(th1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    filtered = [filter_contour_scrollable_area(c, scrollbar_contour, im) for c in contours]
    return [c for c in filtered if c is not None]


def _likely_to_same_contour(contour0, contour1):
    x0, y0, w0, h0 = cv2.boundingRect(contour0)
    x1, y1, w1, h1 = cv2.boundingRect(contour1)
    threshold = 3
    if abs(x1 - x0) > threshold:
        return False
    elif abs(y1 - y0) > threshold:
        return False
    elif abs(w1 - w0) > threshold:
        return False
    elif abs(h1 - h0) > threshold:
        return False
    return True


def _imwrite_debug(im, filename, suffix):
    base, ext = os.path.splitext(filename)
    name = f"{base}_{suffix}{ext}"
    cv2.imwrite(name, im)


def _try_to_detect_scrollbar(im_gray, im_orig_for_debug=None, debug_image_name="", **kwargs):
    """
        スクロールバーおよびスクロール可能領域の検出

        debug 画像を出力したい場合は im_orig_for_debug に二値化
        される前の元画像 (crop されたもの) を渡すこと。
    """
    # 二値化の閾値を高めにするとスクロールバー本体の領域を検出できる。
    # 低めにするとスクロールバー可能領域を検出できる。
    threshold_for_actual = 65
    # スクロール可能領域の判定は、単一の閾値ではどうやっても PNG/JPEG の
    # 両方に対応するのが難しい。そこで、閾値にレンジを設けて高い方から順に
    # トライしていく。閾値が低くなるほど検出されやすいが、矩形がゆがみ
    # やすくなり、後の誤検出につながる。そのため、高い閾値で検出できれば
    # それを正とするのがよい。
    thresholds_for_entire = range(27, 15, -1)

    actual_scrollbar_contours, not_scrollbar_contours = _detect_scrollbar_region(im_gray, threshold_for_actual)
    if im_orig_for_debug is not None and debug_image_name:
        cv2.drawContours(im_orig_for_debug, not_scrollbar_contours, -1, (0.255, 64), 2)
        _imwrite_debug(debug_image_name, im_orig_for_debug, "not_scrollbar")

    if len(actual_scrollbar_contours) == 0:
        return (None, None)

    if im_orig_for_debug is not None:
        cv2.drawContours(im_orig_for_debug, actual_scrollbar_contours, -1, (0, 255, 0), 3)
        _imwrite_debug(debug_image_name, im_orig_for_debug, "scrollbar")

    if len(actual_scrollbar_contours) > 1:
        n = len(actual_scrollbar_contours)
        raise TooManyAreasDetectedError(f'{n} actual scrollbar areas are detected')

    actual_scrollbar_contour = actual_scrollbar_contours[0]

    scrollable_area_contour = None
    for th in thresholds_for_entire:
        scrollable_area_contours = _detect_scrollable_area(im_gray, th, actual_scrollbar_contour)

        if len(scrollable_area_contours) == 0:
            logger.debug(f'th {th}: scrollbar was found, but scrollable area is not found, retry')
            continue

        if len(scrollable_area_contours) > 1:
            if im_orig_for_debug is not None:
                cv2.drawContours(im_orig_for_debug, scrollable_area_contours, -1, (255, 0, 0), 3)
                _imwrite_debug(debug_image_name, im_orig_for_debug, "scrollable_areas")

            n = len(scrollable_area_contours)
            raise TooManyAreasDetectedError(f'{n} scrollable areas are detected')

        scrollable_area_contour = scrollable_area_contours[0]
        same_contour = _likely_to_same_contour(actual_scrollbar_contour, scrollable_area_contour)
        if same_contour:
            # 同じ領域を検出してしまっている場合、誤検出とみなして
            # 閾値を下げてリトライする
            logger.debug(f'th {th}: seems to detect scrollbar as scrollable area, retry')
            continue
        break

    if im_orig_for_debug is not None and scrollable_area_contour is not None:
        cv2.drawContours(im_orig_for_debug, [scrollable_area_contour], -1, (255, 0, 0), 3)
        _imwrite_debug(debug_image_name, im_orig_for_debug, "scrollable_area")

    # thresholds_for_entire のすべての閾値でスクロール可能領域が検出できない
    # 場合は、そもそも元のスクロールバーが誤認識であった可能性が出てくる。
    # この場合 scrollable_area_contour は None になるが、その場合は呼び出し
    # 側でスクロールバー誤検出とみなすようにする。
    return actual_scrollbar_contour, scrollable_area_contour


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
    # 縦4分割して4領域に分け、一番右の領域だけ使う。
    # スクロールバーの領域を調べたいならそれで十分。
    im_h, im_w = im.shape[:2]
    # 縦横比率が規定値を超える場合は上下カットが必要と判断する。
    if im_h / im_w > 0.57:
        cut_size = int(math.ceil(int(im_h - im_w * 0.56) / 2))
        top = cut_size
        bottom = im_h - cut_size
    else:
        cut_size = 0
        top = 0
        bottom = im_h

    logger.debug('top, bottom = (%s, %s), cut_size = %s', top, bottom, cut_size)
    cropped = im[top:bottom, int(im_w*3/4):im_w]
    cr_h, cr_w = cropped.shape[:2]
    logger.debug('cropped image size (for scrollbar): (width, height) = (%s, %s)', cr_w, cr_h)
    im_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    if debug_draw_image:
        im_orig_for_debug = cropped
    else:
        im_orig_for_debug = None

    try:
        actual_scrollbar_region, scrollable_area_region = \
            _try_to_detect_scrollbar(im_gray, im_orig_for_debug, debug_image_name=debug_image_name, **kwargs)
    finally:
        if debug_draw_image:
            logger.debug('writing debug image: %s', debug_image_name)
            cv2.imwrite(debug_image_name, cropped)

    if actual_scrollbar_region is None or scrollable_area_region is None:
        # スクロールバーが検出できない or スクロールバー誤検出（と推定）
        # どちらの場合もスクロールバーなしとして扱う。
        return NOSCROLL_PAGE_INFO

    _, asr_y, _, asr_h = cv2.boundingRect(actual_scrollbar_region)
    _, esr_y, _, esr_h = cv2.boundingRect(scrollable_area_region)

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
