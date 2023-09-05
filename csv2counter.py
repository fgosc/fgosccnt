#!/usr/bin/env python3
# スクショ集計したCSVから周回カウンタ書式に変換
import csv
import sys
import argparse
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Any, List, Dict


logger = logging.getLogger(__name__)

ROW_ITEM_START = 4
MAX_WARNING_COUNT = 5  # 警告の最大文字数

@dataclass
class MaterialLine:
    material: str
    initial: int


@dataclass
class QuestReport:
    questname: str
    runs: int
    lines: List[MaterialLine] = field(default_factory=list)
    note: str = ""

    def __post_init__(self):
        self.update_note()

    def update_note(self):
        """linesの入力内容に応じてnoteの内容を更新する"""
        boosted_appearance_items = [
            "最中",  # 鎌倉イベ
            "団子",  # 鎌倉イベ
            "煎餅",  # 鎌倉イベ
            "クロック",  # 事件簿
            "ラビット",  # 事件簿
            "リーブス",  # 事件簿
        ]
        additional_drop_items = ["宝箱金", "宝箱銀", "宝箱銅"]
        enhanced_drop_items = [
            "根",  # 8502
            "逆鱗",  # 8501
            "心臓",  # 8500
            "涙石",  # 8402
            "勲章",  # 8309
            "貝殻",  # 8308
            "蛇玉",  # 8307
            "羽根",  # 8306
            "蹄鉄",  # 8305
            "ホムベビ",  # 8304
            "頁",  # 8303
            "歯車",  # 8302
            "八連",  # 8301
            "ランタン",  # 8300
            "種",  # 8203
            "毒針",  # 8202
            "塵",  # 8201
            "牙",  # 8200
            "火薬",  # 8105
            "鉄杭",  # 8104
            "髄液",  # 8103
            "鎖",  # 8102
            "骨",  # 8101
            "証",  # 8100
        ]

        for line in self.lines:
            boosted_appearance_rate = (
                "追加出現率 %\n" if line.material in boosted_appearance_items else ""
            )
            self.note += boosted_appearance_rate

        for line in self.lines:
            additional_drop_rate = (
                "追加ドロップ率 %\n" if line.material in additional_drop_items else ""
            )
            self.note += additional_drop_rate

        for line in self.lines:
            enhanced_drop_rate = (
                f"{line.material}泥UP %\n"
                if line.material in enhanced_drop_items
                else ""
            )
            self.note += enhanced_drop_rate

    def set_lines(self, material_lines: List[MaterialLine]):
        """入力メソッドとして使用することにより、noteの内容を更新する

        Args:
            material_lines (List[MaterialLine]): アイテム行
        """
        self.lines = material_lines
        self.update_note()

    def to_fgo_syukai_counter_format(self) -> str:
        """FGO周回カウンタフォーマットに変換する

        Returns:
            str: FGO周回カウンタフォーマットの出力
        """
        formatted_output = f"【{self.questname}】{self.runs}周\n"
        for line in self.lines:
            formatted_output += f"{line.material}{line.initial}\n"

        formatted_output += "#FGO周回カウンタ https://fgodrop.max747.org/\n"
        if len(self.note) > 0:
            formatted_output += "\n" + self.note
        return formatted_output

    def to_dict(self) -> Dict[str, Any]:
        """dictで出力する

        Returns:
            Dict[str, Any]: dict形式の出力
        """
        report_data = asdict(self)
        report_data["lines"] = [asdict(line) for line in self.lines]
        return report_data


def _detect_warnings(lines: List[Dict[str, str]]) -> List[str]:
    messages = []
    # 報酬QP数でエラーチェック
    reward_qp = sum(x.startswith("報酬QP(") for x in lines[0].keys())
    if reward_qp > 1:
        messages.append(f"少なくとも{reward_qp}つのクエストの結果が混在しています")

    for i, item in enumerate(lines):
        if item["filename"] == "missing":
            messages.append(f"{i + 2}行目に missing (複数ページの撮影抜け)があります")
        elif item["filename"].endswith("not valid"):
            messages.append(f"{i + 2}行目に not valid (認識エラー)があります")
        elif item["filename"].endswith("not found"):
            messages.append(f"{i + 2}行目に not found なスクショがあります")
        elif item["filename"].endswith("duplicate"):
            messages.append(f"{i + 2}行目に直前と重複したスクショがあります")

    return messages


def make_warning_message(warning_messages: List[str]) -> str:
    """出力用の警告メッセージを作成する

    Args:
        lines (List[str]): 警告メッセージのリスト

    Returns:
        str: 警告メッセージ
    """
    warning = "\n".join(warning_messages)
    return f"""
###############################################
# WARNING: この処理には以下のエラーがあります #
# 確認せず結果をそのまま使用しないでください #
###############################################
{warning}
###############################################""".lstrip()


def parse_lap_report(lines: List[Dict[str, str]]) -> QuestReport:
    """CSVを入力にしてQuestReportを作成する

    Args:
        lines (List[Dict[str, str]]): CSV入力

    Raises:
        Exception: 入力がおかしいとき

    Returns:
        QuestReport: 初期化された周回情報
    """
    if lines[0]["filename"] != "合計" and len(lines) > 1:
        questname = lines[0]["filename"]
    else:
        if len(lines[1]) <= 3:
            logger.error("認識できるファイル(.JPG, .JPEG, or .PNG)がありません")
            raise Exception
        questname = "周回場所"

    # 周回数出力
    for i, item in enumerate(lines[0].keys()):
        if i == ROW_ITEM_START:
            runs = int(lines[0][item])
            break
    return QuestReport(questname, runs)


def dict_to_material_lines(material_dict: Dict[str, str]) -> List[MaterialLine]:
    """アイテムの辞書をMaterialLineのlistに変換する

    Args:
        material_dict (Dict[str, int]): アイテムの辞書

    Returns:
        List[MaterialLine]: MaterialLineのlist
    """
    excluded_keys = {"filename", "ドロ数", "アイテム数", "獲得QP合計"}

    material_lines = [
        MaterialLine(material=material, initial=int(initial))
        for material, initial in material_dict.items()
        if material not in excluded_keys
        and not material.startswith("報酬QP")
        and initial != ""
    ]

    return material_lines


def main(args):
    sys.stdin = open(sys.stdin.fileno(), "r", encoding="utf_8_sig")
    with args.infile as f:
        reader = csv.DictReader(f)
        lines = [row for row in reader]

    report = parse_lap_report(lines)
    report.set_lines(dict_to_material_lines(lines[0]))

    warning_messages = _detect_warnings(lines)

    if args.json:
        d = report.to_dict()
        if warning_messages:
            d["has_warnings"] = True
            d["warning_messages"] = warning_messages[:MAX_WARNING_COUNT]
            if len(warning_messages) > MAX_WARNING_COUNT:
                d["warning_messages"] += ["(後略)"]
        else:
            d["has_warnings"] = False
        print(json.dumps(d, indent=2, ensure_ascii=False))
        return

    formatted_output = report.to_fgo_syukai_counter_format()
    if warning_messages:
        warning = make_warning_message(warning_messages)
        print(warning)
    print(formatted_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "infile",
        nargs="?",
        type=argparse.FileType("r", encoding="utf_8_sig"),
        default=sys.stdin,
    )
    parser.add_argument("-j", "--json", action="store_true")
    parser.add_argument("-l", "--loglevel", choices=("debug", "info"), default="info")
    args = parser.parse_args()
    lformat = "[%(levelname)s] %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=lformat,
    )
    logger.setLevel(args.loglevel.upper())

    main(args)
