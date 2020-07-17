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
import logging
import os
import sys

import cv2

logger = logging.getLogger('fgo')

NOSCROLL_PAGE_INFO = (1, 1, 0)


class PageInfoError(Exception):
    pass


class CannotGuessError(PageInfoError):
    pass


class TooManyAreasDetectedError(PageInfoError):
    pass


class ScrollableAreaNotFoundError(PageInfoError):
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


def filter_contour_scrollbar(contour, im):
    """
        スクロールバー領域を拾い、それ以外を除外するフィルター
    """
    im_h, im_w = im.shape[:2]
    # 画像全体に対する検出領域の面積比が一定以上であること。
    # 明らかに小さすぎる領域はここで捨てる。
    if cv2.contourArea(contour) * 80 < im_w * im_h:
        return False
    x, y, w, h = cv2.boundingRect(contour)
    logger.debug('scrollbar candidate: (x, y, width, height) = (%s, %s, %s, %s)', x, y, w, h)
    # 縦長領域なので、幅に対して十分大きい高さになっていること。
    if h < w * 5:
        return False
    logger.debug('scrollbar region: (x, y, width, height) = (%s, %s, %s, %s)', x, y, w, h)
    return True


def filter_contour_scrollable_area(contour, im):
    """
        スクロール可能領域を拾い、それ以外を除外するフィルター
    """
    im_w, im_h = im.shape[:2]
    # 画像全体に対する検出領域の面積比が一定以上であること。
    # 明らかに小さすぎる領域はここで捨てる。
    if cv2.contourArea(contour) * 50 < im_w * im_h:
        return False
    x, y, w, h = cv2.boundingRect(contour)
    logger.debug('scrollable area candidate: (x, y, width, height) = (%s, %s, %s, %s)', x, y, w, h)
    # 縦長領域なので、幅に対して十分大きい高さになっていること。
    if h < w * 10:
        return False
    logger.debug('scrollable area region: (x, y, width, height) = (%s, %s, %s, %s)', x, y, w, h)
    return True


def detect_qp_region(im, debug_draw_image=False, debug_image_name=None):
    """
        "所持 QP" 領域を検出する
    """
    # 縦横2分割して4領域に分け、左下の領域だけ使う。
    # QP の領域を調べたいならそれで十分。
    im_h, im_w = im.shape[:2]
    cropped = im[int(im_h/2):im_h, 0:int(im_w/2)]
    cr_h, cr_w = cropped.shape[:2]
    logger.debug('cropped image size (for qp): (width, height) = (%s, %s)', cr_w, cr_h)
    im_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    binary_threshold = 25
    ret, th1 = cv2.threshold(im_gray, binary_threshold, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(th1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    filtered_contours = [c for c in contours if filter_contour_qp(c, im_gray)]
    if len(filtered_contours) == 1:
        qp_region = filtered_contours[0]
        x, y, w, h = cv2.boundingRect(qp_region)
        # 左右の無駄領域を除外する。
        # 感覚的な値ではあるが 左 12%, 右 7% を除外。
        topleft = (x + int(w*0.12), y)
        bottomright = (topleft[0] + w - int(w*0.12) - int(w*0.07), y + h)

        # TODO 切り出した領域をどう扱うか決めてない

        if debug_draw_image:
            cv2.rectangle(cropped, topleft, bottomright, (0, 0, 255), 3)

    if debug_draw_image:
        cv2.drawContours(cropped, filtered_contours, -1, (0, 255, 0), 3)
        logger.debug('writing debug image: %s', debug_image_name)
        cv2.imwrite(debug_image_name, cropped)


def guess_pages(actual_width, actual_height, entire_width, entire_height):
    """
        スクロールバー領域の高さからドロップ枠が何ページあるか推定する
    """
    delta = abs(entire_width - actual_width)
    if delta > 9:
        # 比較しようとしている領域が異なる可能性が高い
        raise CannotGuessError(
            f'幅の誤差が大きすぎます: delta = {delta}, '
            f'entire_width = {entire_width}, '
            f'actual_width = {actual_width}'
        )

    if actual_height * 1.1 > entire_height:
        return 1
    if actual_height * 2.2 > entire_height:
        return 2
    # 4 ページ以上 (ドロップ枠総数 > 63) になることはないと仮定。
    return 3


def guess_pagenum(actual_x, actual_y, entire_x, entire_y, entire_height):
    """
        スクロールバー領域の y 座標の位置からドロップ画像のページ数を推定する
    """
    if abs(actual_x - entire_x) > 9:
        # 比較しようとしている領域が異なる可能性が高い
        raise CannotGuessError(f'x 座標の誤差が大きすぎます: entire_x = {entire_x}, actual_x = {actual_x}')

    # スクロールバーと上端との空き領域の縦幅 delta と
    # スクロール可能領域の縦幅 entire_height との比率で位置を推定する。
    delta = actual_y - entire_y
    ratio = delta / entire_height
    logger.debug('space above scrollbar: %s, entire_height: %s, ratio: %s', delta, entire_height, ratio)
    if ratio < 0.1:
        return 1
    # 実測では 0.47-0.50 の間くらいになる。
    # 7列3ページの3ページ目の値が 0.55 近辺なので、あまり余裕を持たせて大きくしすぎてもいけない。
    # このあたりから 0.52 くらいが妥当な線ではないか。
    if ratio < 0.52:
        return 2
    # 4 ページ以上になることはないと仮定。
    return 3


def guess_lines(actual_width, actual_height, entire_width, entire_height):
    """
        スクロールバー領域の高さからドロップ枠が何行あるか推定する
        スクロールバーを用いる関係上、原理的に 2 行以下は推定不可
    """
    delta = abs(entire_width - actual_width)
    if delta > 9:
        # 比較しようとしている領域が異なる可能性が高い
        raise CannotGuessError(
            f'幅の誤差が大きすぎます: delta = {delta}, '
            f'entire_width = {entire_width}, '
            f'actual_width = {actual_width}'
        )

    ratio = actual_height / entire_height
    logger.debug('scrollbar ratio: %s', ratio)
    if ratio > 0.90:    # 実測値 0.94
        return 3
    elif ratio > 0.70:  # 実測値 0.72-0.73
        return 4
    elif ratio > 0.57:  # 実測値 0.59-0.60
        return 5
    elif ratio > 0.48:  # 実測値 0.50-0.51
        return 6
    elif ratio > 0.40:  # サンプルなし 参考値 1/2.333 = 0.429, 1/2.5 = 0.4
        return 7
    elif ratio > 0.36:  # サンプルなし 参考値 1/2.666 = 0.375, 1/2.77 = 0.361
        return 8
    else:
        # 10 行以上は考慮しない
        return 9


def _detect_scrollbar_region(im, binary_threshold, filter_func):
    ret, th1 = cv2.threshold(im, binary_threshold, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(th1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if filter_func(c, im)]


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


def _try_to_detect_scrollbar(im_gray, im_orig_for_debug=None, **kwargs):
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
    thresholds_for_entire = (27, 26, 25, 24, 23)

    actual_scrollbar_contours = _detect_scrollbar_region(
        im_gray, threshold_for_actual, filter_contour_scrollbar)
    if len(actual_scrollbar_contours) == 0:
        return (None, None)

    if im_orig_for_debug is not None and kwargs.get('draw_greenline'):
        cv2.drawContours(im_orig_for_debug, actual_scrollbar_contours, -1, (0, 255, 0), 3)

    if len(actual_scrollbar_contours) > 1:
        n = len(actual_scrollbar_contours)
        raise TooManyAreasDetectedError(f'{n} actual scrollbar areas are detected')

    actual_scrollbar_contour = actual_scrollbar_contours[0]

    scrollable_area_contour = None
    for th in thresholds_for_entire:
        scrollable_area_contours = _detect_scrollbar_region(
            im_gray, th, filter_contour_scrollable_area)
        if len(scrollable_area_contours) == 0:
            logger.debug(f'th {th}: scrollbar was found, but scrollable area is not found, retry')
            continue

        if len(scrollable_area_contours) > 1:
            if im_orig_for_debug is not None and kwargs.get('draw_blueline'):
                cv2.drawContours(im_orig_for_debug, scrollable_area_contours, -1, (255, 0, 0), 3)

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

    if im_orig_for_debug is not None \
        and scrollable_area_contour is not None \
            and kwargs.get('draw_blueline'):
        cv2.drawContours(im_orig_for_debug, [scrollable_area_contour], -1, (255, 0, 0), 3)

    # thresholds_for_entire のすべての閾値でスクロール可能領域が検出できない
    # 場合は、そもそも元のスクロールバーが誤認識であった可能性が出てくる。
    # この場合 scrollable_area_contour は None になるが、その場合は呼び出し
    # 側でスクロールバー誤検出とみなすようにする。
    return actual_scrollbar_contour, scrollable_area_contour


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
    cropped = im[0:im_h, int(im_w*3/4):im_w]
    cr_h, cr_w = cropped.shape[:2]
    logger.debug('cropped image size (for scrollbar): (width, height) = (%s, %s)', cr_w, cr_h)
    im_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    if debug_draw_image:
        im_orig_for_debug = cropped
    else:
        im_orig_for_debug = None

    try:
        actual_scrollbar_region, scrollable_area_region = \
            _try_to_detect_scrollbar(im_gray, im_orig_for_debug, **kwargs)
    finally:
        if debug_draw_image:
            logger.debug('writing debug image: %s', debug_image_name)
            cv2.imwrite(debug_image_name, cropped)

    if actual_scrollbar_region is None or scrollable_area_region is None:
        # スクロールバーが検出できない or スクロールバー誤検出（と推定）
        # どちらの場合もスクロールバーなしとして扱う。
        return NOSCROLL_PAGE_INFO

    asr_x, asr_y, asr_w, asr_h = cv2.boundingRect(actual_scrollbar_region)
    esr_x, esr_y, esr_w, esr_h = cv2.boundingRect(scrollable_area_region)

    pages = guess_pages(asr_w, asr_h, esr_w, esr_h)
    pagenum = guess_pagenum(asr_x, asr_y, esr_x, esr_y, esr_h)
    lines = guess_lines(asr_w, asr_h, esr_w, esr_h)
    return (pagenum, pages, lines)


def look_into_file(filename, args):
    logger.debug(f'===== {filename}')

    im = cv2.imread(filename)
    if im is None:
        raise FileNotFoundError(f'Cannot read file: {filename}')

    im_h, im_w = im.shape[:2]
    logger.debug('image size: (width, height) = (%s, %s)', im_w, im_h)

    # TODO QP 領域をどう扱うか未定
    # if args.debug_qp:
    #     debug_qp_dir = os.path.join(args.debug_out_dir, 'qp')
    #     os.makedirs(debug_qp_dir, exist_ok=True)
    #     debug_qp_image = os.path.join(debug_qp_dir, os.path.basename(filename))
    # else:
    #     debug_qp_image = None
    # detect_qp_region(im, args.debug_qp, debug_qp_image)

    if args.debug_sc:
        debug_sc_dir = os.path.join(args.debug_out_dir, 'sc')
        os.makedirs(debug_sc_dir, exist_ok=True)
        debug_sc_image = os.path.join(debug_sc_dir, os.path.basename(filename))
    else:
        debug_sc_image = None
    kwargs = {
        'draw_greenline': not args.debug_disable_greenline,
        'draw_blueline': not args.debug_disable_blueline,
    }
    pagenum, pages, lines = guess_pageinfo(im, args.debug_sc, debug_sc_image, **kwargs)
    logger.debug('pagenum: %s, pages: %s, lines: %s', pagenum, pages, lines)
    return (pagenum, pages, lines)


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
    parser.add_argument('filename', nargs='+')
    parser.add_argument(
        '-l', '--loglevel',
        choices=('debug', 'info', 'warning'),
        default='info',
        help='set loglevel [default: info]',
    )
    # parser.add_argument(
    #     '-dq', '--debug-qp',
    #     action='store_true',
    #     help='enable writing qp image for debug',
    # )
    parser.add_argument(
        '-ds', '--debug-sc',
        action='store_true',
        help='enable writing sc image for debug',
    )
    parser.add_argument(
        '--debug-disable-blueline',
        action='store_true',
        help='disable drawing blue line on sc image for debug',
    )
    parser.add_argument(
        '--debug-disable-greenline',
        action='store_true',
        help='disable drawing green line on sc image for debug',
    )
    parser.add_argument(
        '-do', '--debug-out-dir',
        default='debugimages',
        help='output directory for debug images [default: debugimages]',
    )
    parser.add_argument(
        '-o', '--output',
        type=argparse.FileType('w'),
        default=sys.stdout,
        help='output file [default: STDOUT]',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    logger.setLevel(args.loglevel.upper())
    main(args)
