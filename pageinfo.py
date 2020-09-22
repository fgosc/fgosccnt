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
import os
import sys

import cv2

logger = logging.getLogger('fgo')

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
        # 感覚的な値ではあるが 左 42%, 右 4% を除外。
        # 落とし穴として、2019年5月末 ～ 9月の間に所持 QP の出力位置が微妙に変わっている。
        # ここではそのどちらのケースでも対応できるよう枠を広めに取っている。
        # 現仕様に最適化して切り詰めすぎると困ったことになるため注意。
        left_margin = 0.42
        right_margin = 0.04

    if len(filtered_contours) == 1:
        qp_region = filtered_contours[0]
        x, y, w, h = cv2.boundingRect(qp_region)

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


def guess_pages(actual_width, actual_height, entire_width, entire_height):
    """
        スクロールバー領域の高さからドロップ枠が何ページあるか推定する
    """
    delta = abs(entire_width - actual_width)
    if delta > 15:
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
    if abs(actual_x - entire_x) > 12:
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
    if delta > 15:
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
    elif ratio > 0.42:  # 実測値 0.44
        return 7
    elif ratio > 0.38:  # 実測値 0.40-0.41
        return 8
    else:
        # 10 行以上は考慮しない
        return 9


def _detect_scrollbar_region(im, binary_threshold, filter_func):
    _, th1 = cv2.threshold(im, binary_threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(th1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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
    thresholds_for_entire = range(27, 17, -1)

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


def look_into_file_for_page(filename, im, args):
    if args.debug_sc:
        debug_sc_dir = os.path.join(args.debug_out_dir, 'page')
        os.makedirs(debug_sc_dir, exist_ok=True)
        prefix = args.debug_out_file_prefix
        debug_image = os.path.join(debug_sc_dir, prefix + os.path.basename(filename))
        logger.debug('debug image path: %s', debug_image)
    else:
        debug_image = None
    kwargs = {
        'draw_greenline': not args.debug_disable_greenline,
        'draw_blueline': not args.debug_disable_blueline,
    }
    pagenum, pages, lines = guess_pageinfo(im, args.debug_sc, debug_image, **kwargs)
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
    page_parser.add_argument(
        '--debug-disable-blueline',
        action='store_true',
        help='disable drawing blue line on sc image for debug',
    )
    page_parser.add_argument(
        '--debug-disable-greenline',
        action='store_true',
        help='disable drawing green line on sc image for debug',
    )
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
