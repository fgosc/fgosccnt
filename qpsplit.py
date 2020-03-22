import fgosccnt2
import argparse
import cv2
from pathlib import Path
import shutil

training = Path(__file__).resolve().parent / Path("training.xml")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FGOスクショからアイテムをCSV出力する')
    parser.add_argument('filenames', help='入力ファイル', nargs='*')    # 必須の引数を追加
    args = parser.parse_args()    # 引数を解析
    svm = cv2.ml.SVM_load(str(training))

    prev_pagenum = 0
    prev_chestnum = 0
    
    for filename in args.filenames:
        f = Path(filename)
        if f.exists() == False:
            print( filename + 'is not found.')
        else:            
            img_rgb = fgosccnt2.imread(filename)
            a = fgosccnt2.ScreenShot(img_rgb, svm)
##            print(a.qplist)
##            print(a.chestnum)
##            print(a.pagenum)
            if a.reward != "":
                Qp_dir = Path(a.reward)
                if not Qp_dir.is_dir():
                    Qp_dir.mkdir()
                dstfile = Qp_dir / Path(f).name
                new_path = shutil.move(Path(f), Path(dstfile))
##                print(new_path)
                prev_pagenum = a.pagenum
                prev_chestnum = a.chestnum
            else:
                if prev_chestnum == a.chestnum and prev_pagenum == a.pagenum - 1:
                    dstfile = Qp_dir / Path(f).name
                    new_path = shutil.move(Path(f), Path(dstfile))
