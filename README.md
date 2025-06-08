![fgosccnt1](https://user-images.githubusercontent.com/62515228/78866947-437c3100-7a7b-11ea-8eb7-7771786b1763.png)

# fgosccnt

FGO のバトルリザルトのスクリーンショットから戦利品をカウント

このプログラムをインストールしなくても Web 版でお試しできます(詳しくない人にはこちらを推奨)
https://github.com/fgophi/fgosccnt/wiki/Easy-Use

# 必要なソフトウェア

1. Python 3.7 以降
2. Tesseract OCR
   - Mac, Linx 等: https://github.com/tesseract-ocr/tesseract
   - Windows: https://github.com/UB-Mannheim/tesseract/wiki

# ファイル

1. fgosccnt.py :実行ファイル
2. makeitem.py item.xml を作成
3. makechest.py chest.xml を作成
4. makecard.py card.xml を作成
5. makedcnt.py dcnt.xml を生成
6. data フォルダ 2.3.4.5.で用いられるファイル
7. csv2counter.py (おまけ)fgosccnt.py の出力 CSV を FGO 周回カウンタ書式にする
8. qpsplit.py (おまけ)スクショファイルを報酬 QP ごとにフォルダ分けする

以下は 2.3.4.実行時に作成される

9. item.xml: アイテム下部の文字を読む SVM のトレーニングファイル
10. chest.xml: 旧 UI のドロップ数の文字を読む SVM のトレーニングファイル
11. card.xml: カード下部の文字を読む SVM のトレーニングファイル
12. dcnt.xml: 新 UI のドロップ数の文字を読む SVM のトレーニングファイル
13. exp_class.xml: 種火のクラス判別をする SVM のトレーニングファイル

# インストール

- OpenCV をインストール

```
$ pip install -r requirements.txt
```

- Tesseract OCR をインストール
- fgoscdata を使用できるようにする (submodule の初期化)

```
$ git submodule update --init
```

- makeitem.py, makechest.py, makecard.py をそれぞれ実行

下記コマンドを実行

```
$ python makeitem.py
$ python makechest.py
$ python makecard.py
$ python makedcnt.py
$ python makeexpcls.py
```

※fgosccnt.py, item.xml chest.xml card.xml dcnt.xml を同じフォルダにいれること

# 使い方

```
usage: fgosccnt.py [-h] [-f FOLDER] [-t TIMEOUT]
                   [--ordering {notspecified,filename,timestamp}] [-d]
                   [--version]
                   [filenames [filenames ...]]

FGOスクショからアイテムをCSV出力する

positional arguments:
  filenames             入力ファイル

optional arguments:
  -h, --help            show this help message and exit
  -f FOLDER, --folder FOLDER
                        フォルダで指定
  -t TIMEOUT, --timeout TIMEOUT
                        QPカンスト時の重複チェック感覚(秒): デフォルト15秒
  --ordering {notspecified,filename,timestamp}
                        ファイルの処理順序 (未指定の場合 notspecified)
  -d, --debug           デバッグ情報の出力
  --version             show program's version number and exit
```

# 実行結果

    $ python fgosccnt.py ファイル1 ファイル2... > output.csv

## 結果(例 1)

    filename,宝箱数,item000001(+1900),item000002(x3),item000003(x3),item000004
    合計,26,3,3,22,1
    IMG_2065.PNG,10,1,2,8,
    IMG_2066.PNG,9,1,1,7,1
    IMG_2067.PNG,7,1,,7,

## 結果(例 2):item ファイルの名前を変えた場合

    filename,宝箱数,QP(+1900),薬草(x3),麦袋(x3),騎ピ
    合計,26,3,3,22,1
    IMG_2065.PNG,10,1,2,8,
    IMG_2066.PNG,9,1,1,7,1
    IMG_2067.PNG,7,1,,7,

---

- 日本版 FGO に加え NA 版 FGO のバトルリザルトのスクショにも対応している
- PNG に加え、JPEG ファイルにも対応した(テストは PNG に比べて十分に行えていない)
  - JPEG の品質を落とすと当然認識ミスが増えます 55 で確認した例あり
  - おそらく OpenCV が対応しているフォーマットのファイルなら扱えるはず
- 恒常アイテム名とポイントは自動認識される
- イベント限定アイテムを初めて認識させた場合、item フォルダ内に item*.png というファイルができる
  　　* item\*.png のファイル名をアイテム名に変更すると次回実行以降もそのアイテム名で表示されるため例 2 のように出力され可読性があがる
- 戦利品下部に数字が書かれている場合、その数字を認識して出力する(ただしボーナス表記は省かれる)
  - そのため例えば扇 x2 と扇 x3 は同じ item ファイル 1 つで認識される
- アイテムが別のものと誤認識される場合、個別のカードファイルを作って item/ にいれればそのカード名で認識される
  - 個別のカードファイルは -d オプションでスクショを読み込ませればできる
- 全く同じアイテムで別ファイルができる場合があるが、フォルダを分けて同じ名前のアイテムファイルにすれば同じものとしてカウントされる
- 複数解像度の読み込みに対応している(極端な低解像度のテストは十分に行えていない)
- 同じ戦闘結果のスクショが検知された場合は、file 名: duplicate と出力されアイテム数は出ない
  - (QP カンストしていない場合)ドロップアイテムが同じで QP が同じ場合
  - (QP カンストしている場合)ドロップアイテムが同じでファイルの EXIF データの作成日時の差が 15 秒未満の場合(秒数は-t オプションで変更可能)

# 制限

- 全く同じアイテムでも複数のアイテムファイルが作成されることがごく稀にある
- 認識率は 100%ではない(低解像度なほど問題が起きやすい)
- QP 画像の item???.png ができてしまったとき、取り除かないと QP が末尾にくるという順番ルールが守られない
