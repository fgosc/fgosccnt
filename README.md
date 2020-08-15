![fgosccnt1](https://user-images.githubusercontent.com/62515228/78866947-437c3100-7a7b-11ea-8eb7-7771786b1763.png)
# fgosccnt
FGOのバトルリザルトのスクリーンショットから戦利品をカウント

このプログラムをインストールしなくてもWeb版でお試しできます(詳しくない人にはこちらを推奨)
https://github.com/fgophi/fgosccnt/wiki/Easy-Use

# 必要なソフトウェア
1. Python 3.7以降
2. Tesseract OCR
   - Mac, Linx 等: https://github.com/tesseract-ocr/tesseract
   - Windows: https://github.com/UB-Mannheim/tesseract/wiki

# ファイル
1. fgosccnt.py :実行ファイル
2. makeitem.py item.xml を作成
3. makechest.py chest.xml を作成
4. makecard.py card.xml を作成
5. data フォルダ 2.3.4.で用いられるファイル
6. csv2counter.py (おまけ)fgosccnt.pyの出力CSVをFGO周回カウンタ書式にする
7. qpsplit.py (おまけ)スクショファイルを報酬QPごとにフォルダ分けする

以下は2.3.4.実行時に作成される

8. item.xml: アイテム下部の文字を読むSVMのトレーニングファイル
9. chest.xml:  ドロップ数の文字を読むSVMのトレーニングファイル
10. card.xml:  カード下部の文字を読むSVMのトレーニングファイル

# インストール

* OpenCV をインストール
* Tesseract OCR をインストール
* makeitem.py, makechest.py, makecard.py をそれぞれ実行

下記コマンドを実行

    $ python makeitem.py
    $ python makechest.py
    $ python makecard.py

※fgosccnt.py, item.xml chest.xml card.xmlを同じフォルダにいれること


# 使い方

    usage: fgosccnt.py [-h] [--version] [filenames [filenames ...]]
    
    FGOスクショからアイテムをCSV出力する
    
    positional arguments:
      filenames   入力ファイル
    
    optional arguments:
      -h, --help  show this help message and exit
      -d, --debug  デバッグ情報の出力
      --version   show program's version number and exit

# 実行結果
    $ python fgosccnt.py ファイル1 ファイル2... > output.csv

## 結果(例1)
    filename,宝箱数,item000001(+1900),item000002(x3),item000003(x3),item000004
    合計,26,3,3,22,1
    IMG_2065.PNG,10,1,2,8,
    IMG_2066.PNG,9,1,1,7,1
    IMG_2067.PNG,7,1,,7,

## 結果(例2):itemファイルの名前を変えた場合
    filename,宝箱数,QP(+1900),薬草(x3),麦袋(x3),騎ピ
    合計,26,3,3,22,1
    IMG_2065.PNG,10,1,2,8,
    IMG_2066.PNG,9,1,1,7,1
    IMG_2067.PNG,7,1,,7,

***
* 日本版FGOに加えNA版FGOのバトルリザルトのスクショにも対応している
* PNGに加え、JPEGファイルにも対応した(テストはPNGに比べて十分に行えていない)
  * JPEGの品質を落とすと当然認識ミスが増えます 55 で確認した例あり 
  * おそらくOpenCVが対応しているフォーマットのファイルなら扱えるはず
  
* 恒常アイテム名とポイントは自動認識される
* イベント限定アイテムを初めて認識させた場合、item フォルダ内に item*.png というファイルができる
　　* item*.pngのファイル名をアイテム名に変更すると次回実行以降もそのアイテム名で表示されるため例2のように出力され可読性があがる
* 戦利品下部に数字が書かれている場合、その数字を認識して出力する(ただしボーナス表記は省かれる)
  * そのため例えば扇x2と扇x3は同じitemファイル1つで認識される
* アイテムが別のものと誤認識される場合、個別のカードファイルを作って item/ にいれればそのカード名で認識される
  * 個別のカードファイルは -d オプションでスクショを読み込ませればできる
* 全く同じアイテムで別ファイルができる場合があるが、フォルダを分けて同じ名前のアイテムファイルにすれば同じものとしてカウントされる
* 複数解像度の読み込みに対応している(極端な低解像度のテストは十分に行えていない)
* 同じ戦闘結果のスクショが検知された場合は、file名: duplicate と出力されアイテム数は出ない
  * (QPカンストしていない場合)ドロップアイテムが同じでQPが同じ場合
  * (QPカンストしている場合)ドロップアイテムが同じでファイル作成時間が15未満の場合

# 制限
* 全く同じアイテムでも複数のアイテムファイルが作成されることがごく稀にある
* 認識率は100%ではない(低解像度なほど問題が起きやすい)
* QP画像のitem???.pngができてしまったとき、取り除かないとQPが末尾にくるという順番ルールが守られない
