import argparse
import cv2
from pathlib import Path
import shutil
import logging

import fgosccnt

logger = logging.getLogger(__name__)

training = Path(__file__).resolve().parent / Path("item.xml")
train_item = Path(__file__).resolve().parent / Path("item.xml")  # アイテム下部
train_chest = Path(__file__).resolve().parent / Path("chest.xml")  # ドロップ数
train_card = Path(__file__).resolve().parent / Path("card.xml")  # ドロップ数


def file_Assignment(args, files):
    svm = cv2.ml.SVM_load(str(train_item))
    svm_chest = cv2.ml.SVM_load(str(train_chest))
    svm_card = cv2.ml.SVM_load(str(train_card))

    prev_pagenum = 0
    prev_chestnum = 0

    for filename in files:
        f = Path(filename)
        if f.exists() is False:
            print(filename + ' is not found.')
        else:
            img_rgb = fgosccnt.imread(filename)
            fileextention = f.suffix
            try:
                a = fgosccnt.ScreenShot(
                                        args, img_rgb,
                                        svm, svm_chest, svm_card,
                                        fileextention,
                                        reward_only=True
                                        )
            except Exception:
                print(Path(f).name, end=": ")
                print("正常なFGOのバトルリザルトのスクショではありません")
                continue
            if a.itemlist[0]["id"] == fgosccnt.ID_REWARD_QP:
                Qp_dir = Path(
                              "QP" + "(+" + str(
                                                a.itemlist[0]["dropnum"]
                                                ) + ")"
                              )
                if not Qp_dir.is_dir():
                    Qp_dir.mkdir()
                dstfile = Qp_dir / Path(f).name
                shutil.move(Path(f), Path(dstfile))
                print(Path(f).name, end=" => ")
                print(Qp_dir)
                prev_pagenum = a.pagenum
                prev_chestnum = a.chestnum
            else:
                logger.debug("prev_chestnum: %s", prev_chestnum)
                logger.debug("a.chestnum: %s", a.chestnum)
                logger.debug("prev_pagenum: %s", prev_pagenum)
                logger.debug("a.pagenum: %s", a.pagenum)
                if prev_chestnum == a.chestnum \
                   and prev_pagenum == a.pagenum - 1:
                    dstfile = Qp_dir / Path(f).name
                    shutil.move(Path(f), Path(dstfile))
                    print(Path(f).name, end=" => ")
                    print(Qp_dir)
                    prev_pagenum = a.pagenum
                    prev_chestnum = a.chestnum
                else:
                    print(Path(f).name, end=" => ")
                    print("移動無し(1ページ目不明)")
                    prev_pagenum = 0
                    prev_chestnum = 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FGOスクショからアイテムをCSV出力する')
    parser.add_argument('--lang', default=fgosccnt.DEFAULT_ITEM_LANG,
                        choices=('jpn', 'eng'),
                        help='Language to be used for output: Default '
                             + fgosccnt.DEFAULT_ITEM_LANG)
    parser.add_argument('filenames', help='入力ファイル', nargs='*')    # 必須の引数を追加
    parser.add_argument('-l', '--loglevel',
                        choices=('debug', 'info'), default='info')
    args = parser.parse_args()    # 引数を解析
    lformat = '[%(levelname)s] %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=lformat,
    )
    logger.setLevel(args.loglevel.upper())

    file_Assignment(args, args.filenames)
