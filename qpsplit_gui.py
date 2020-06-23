import PySimpleGUI as sg
from os.path import basename
import os
import qpsplit
import configparser
import sys
from pathlib import Path

sg.Print(do_not_reroute_stdout=True)

config = configparser.ConfigParser()
settingfile = Path(__file__).resolve().parent / 'setting.ini'
output_folder = "出力先フォルダ"
try:
    if not settingfile.exists():
        settingfile.touch()
        
    config.read(settingfile)
    section0 = "default"
    if section0 not in config.sections():
        config.add_section(section0)
    section0cfg = config[section0]

    output_folder = section0cfg.get("output_folder", output_folder)


except :
    print("[エラー] setting.iniに不備があります。")
    sys.exit(1)

##print("output_folder:")
##print(output_folder)

col1 = [[sg.Button('実行')],
        [sg.Button('終了')]]


layout = [[sg.Text('ファイル選択', size=(15, 1), justification='right'),
          sg.InputText('ファイル一覧', enable_events=True, key='-FILES-',),
          sg.FilesBrowse('ファイルを追加', file_types=(('PNG ファイル', '*.png'),))],
          
          [sg.Text('出力先フォルダ', size=(15, 1), justification='right'),
          sg.InputText(output_folder, enable_events=True, key='-FOLDER-'),
          sg.FolderBrowse('出力フォルダを変更')],
          [sg.Button('ログをコピー'), sg.Button('ログをクリア')],
          [sg.Output(size=(100, 5), key='-MULTILINE-')],
          [sg.Button('入力一覧をクリア')],
          [sg.Listbox([], size=(100, 10), enable_events=True, key='-LIST-')],
          [sg.Column(col1)]]
##          [sg.Frame('処理内容', frame1), sg.Column(col1)]]

window = sg.Window('バトルリザルトのQPによる振り分け', layout)

new_files = []
new_file_names = []

while True:             # Event Loop
    event, values = window.read()
    if event in (None, '終了'):
        if values['-FOLDER-'] != "":
            config.set(section0, "output_folder", values['-FOLDER-'])
            with open(settingfile, "w") as file:
                config.write(file)
        break

    if event == '実行':
        print('処理を実行')
        print('処理対象ファイル：', new_files)

        if values['-FOLDER-'] != "出力先フォルダ":
            os.chdir(values['-FOLDER-'])
        print("作業フォルダ: ", end="")
        print(os.getcwd())
##        sg.popup(new_files)
        qpsplit.file_Assignment(new_files)

        # ポップアップ
        sg.popup('処理が正常終了しました')
    elif event == 'ログをクリア':
        print('ログをクリア')
        window.FindElement('-MULTILINE-').Update('')
    elif event == 'ログをコピー':
        window.FindElement('-MULTILINE-').Widget.clipboard_append(window.find_element('-MULTILINE-').Get())
        sg.popup('ログをコピーしました')
    elif event == '入力一覧をクリア':
        print('入力一覧をクリア')

        new_files.clear()
        new_file_names.clear()
        window['-LIST-'].update('')
    elif event == '-FILES-':
        print('FilesBrowse')

        # TODO:実運用には同一ファイルかどうかの処理が必要
        for f in values['-FILES-'].split(';'):
            if f not in new_files:
                new_files.append(f)
        new_file_names = [basename(file_path) for file_path in new_files]

        print('ファイルを追加')
        window['-LIST-'].update(new_file_names)  # リストボックスに表示します
    elif event == '-FOLDER-':
        print('FolderChange')
        print(values['-FOLDER-'])

window.close()
