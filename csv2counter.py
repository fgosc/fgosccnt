## スクショ集計したCSVから周回カウンタ書式に変換
import csv
import sys
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('infile', nargs='?', type=argparse.FileType(),
                        default=sys.stdin)
    args = parser.parse_args()

    with args.infile as f:
        reader = csv.DictReader(f)
        l = [row for row in reader]

    print ("【周回場所】", end="")
    output = ""
    for i, item in enumerate(l[0].keys()):
        if i == 2:
            print (l[0][item] + "周")
        if i > 2:
            output =  output + item + l[0][item] + "-"

    print (output[:-1])
    print ("#FGO周回カウンタ http://aoshirobo.net/fatego/rc/")

