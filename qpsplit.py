import fgosccnt
import argparse
import cv2
from pathlib import Path
import shutil

training = Path(__file__).resolve().parent / Path("item.xml")
train_item = Path(__file__).resolve().parent / Path("item.xml") #アイテム下部
train_chest = Path(__file__).resolve().parent / Path("chest.xml") #ドロップ数
train_card = Path(__file__).resolve().parent / Path("card.xml") #ドロップ数

def file_Assignment(files):
    svm = cv2.ml.SVM_load(str(train_item))
    svm_chest = cv2.ml.SVM_load(str(train_chest))
    svm_card = cv2.ml.SVM_load(str(train_card))

    prev_pagenum = 0
    prev_chestnum = 0
    
    for filename in files:
        f = Path(filename)
        if f.exists() == False:
            print( filename + 'is not found.')
        else:            
            img_rgb = fgosccnt.imread(filename)
            fileextention = f.suffix
            try:
                a = fgosccnt.ScreenShot(img_rgb, svm, svm_chest, svm_card,
                                    fileextention, reward_only=True)
            except:
                print(Path(f).name, end=": ")
                print("正常なFGOのバトルリザルトのスクショではありません")
                continue
##            print(a.qplist)
##            print(a.chestnum)
##            print(a.pagenum)
            if a.reward != "":
                Qp_dir = Path(a.reward)
                if not Qp_dir.is_dir():
                    Qp_dir.mkdir()
                dstfile = Qp_dir / Path(f).name
                new_path = shutil.move(Path(f), Path(dstfile))
                print(Path(f).name, end=" => ")
                print(Qp_dir)
##                print(new_path)
                prev_pagenum = a.pagenum
                prev_chestnum = a.chestnum
            else:
                if prev_chestnum == a.chestnum and prev_pagenum == a.pagenum - 1:
                    dstfile = Qp_dir / Path(f).name
                    new_path = shutil.move(Path(f), Path(dstfile))
                    print(Path(f).name, end=" => ")
                    print(Qp_dir)
                else:
                    print(Path(f).name, end=" => ")
                    print("移動無し(1ページ目不明)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FGOスクショからアイテムをCSV出力する')
    parser.add_argument('filenames', help='入力ファイル', nargs='*')    # 必須の引数を追加
    args = parser.parse_args()    # 引数を解析

    file_Assignment(args.filenames)
    
##    svm = cv2.ml.SVM_load(str(train_item))
##    svm_chest = cv2.ml.SVM_load(str(train_chest))
##    svm_card = cv2.ml.SVM_load(str(train_card))
##
##    prev_pagenum = 0
##    prev_chestnum = 0
##    
##    for filename in args.filenames:
##        f = Path(filename)
##        if f.exists() == False:
##            print( filename + 'is not found.')
##        else:            
##            img_rgb = fgosccnt.imread(filename)
##            a = fgosccnt.ScreenShot(img_rgb, svm, svm_chest, svm_card)
####            print(a.qplist)
####            print(a.chestnum)
####            print(a.pagenum)
##            if a.reward != "":
##                Qp_dir = Path(a.reward)
##                if not Qp_dir.is_dir():
##                    Qp_dir.mkdir()
##                dstfile = Qp_dir / Path(f).name
##                new_path = shutil.move(Path(f), Path(dstfile))
####                print(new_path)
##                prev_pagenum = a.pagenum
##                prev_chestnum = a.chestnum
##            else:
##                if prev_chestnum == a.chestnum and prev_pagenum == a.pagenum - 1:
##                    dstfile = Qp_dir / Path(f).name
##                    new_path = shutil.move(Path(f), Path(dstfile))
