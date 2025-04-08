import glob
import json
import openai
import os
import time
import re
import random
import argparse

# OpenAI APIに問い合わせるためのクラス
class Get:
    def __init__(self):
        self.prompt = ""
        
    def calc(self, query, temp=0, n=1, model='gpt-4-1106-preview'):
        # 指定したクエリ（プロンプト）を使ってAPIに問い合わせ、返答と使用トークン数を返す
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "system", "content": query}],
            temperature=temp,
            top_p=0.9,
            n=n
        )
        res = [tem["message"]["content"] for tem in response["choices"]]
        return res, {'prompt': response['usage']['prompt_tokens'],
                     'completion': response['usage']['completion_tokens']}

# レスポンスからスコア群を抽出する関数
def get_score(s_, num=10):
    s_ = s_.lower()
    s = s_.split("scores:")[-1]
    try:
        matches = re.findall(r'\[[^\]]*\]', s)[0]
        s = matches[1:-1].split(",")
        s = [t.split(":")[-1].strip(" ") for t in s]
        for t in range(len(s)):
            try:
                float(s[t])
            except:
                s[t] = '0'
        s = [float(t) for t in s]
    except:
        try:
            s = s_.split("scores:")[-1]
            scores = re.findall(r'sample\s*\d+:\s*([\d\.]+)', s)
            s = [float(t) for t in scores]
        except:
            s = []
    if len(s) != num and len(s) != 5:
        s = [-1 for _ in range(num)]
    return s

# 生成された各バッチの結果ファイルを処理し、新たな評価データを生成する関数
def generate_new_data(dir_name, data_idx, further_dir, batchsize):
    # 指定されたディレクトリから生データファイルを取得
    generate_name = "API_Completion/" + dir_name + "{}.json/{}/raw_data/*".format(data_idx, further_dir)
    dirs = glob.glob(generate_name)
    raw_datas = []
    for file in dirs:
        with open(file, "r") as f:
            l = json.load(f)
        raw_datas.append(l)
    
    # 生データから各サンプルに対してスコアを付与
    datas = []
    for tem in raw_datas:
        score = get_score(tem['response'][0], num=batchsize)
        for i in range(len(tem['raw_data'])):
            tem['raw_data'][i]['modelscore'] = score[i]
            datas.append(tem['raw_data'][i])
    # スコアが高い順にソート
    datas.sort(key=lambda x: x['modelscore'], reverse=True)
    
    # バッチサイズに合わせてサンプルを再配置
    other_idx = random.sample(range(len(datas)), len(datas) % batchsize)
    datas_ = [datas[i] for i in range(len(datas)) if i not in other_idx]
    new_datas = []
    bottle = len(datas_) // batchsize
    for i in range(bottle):
        cur_batch = []
        for j in range(batchsize):
            cur_batch.append(datas_[bottle * j + i])
        random.shuffle(cur_batch)
        new_datas += cur_batch
    for t in other_idx:
        new_datas.append(datas[t])
    
    # バッチ全体の新しいデータをディクショナリ形式にまとめ、ファイルに出力
    new_batch = {}
    for key in new_datas[0].keys():
        new_batch[key] = [t[key] for t in new_datas]
    with open("eval_data_new/" + dir_name + "{}.json".format(data_idx + 1), "w") as f:
        json.dump(new_batch, f)

if __name__ == '__main__':
    # コマンドライン引数の設定
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--iteration_round', type=int, default=5)
    argparser.add_argument('--temperature', type=float, default=0.2)
    argparser.add_argument('--n', type=int, default=1)
    argparser.add_argument("--batchsize", type=int, default=10)
    argparser.add_argument('--model', type=str, default="gpt-4-1106-preview")
    argparser.add_argument('--criterion', type=str, default='Coherent')
    args = argparser.parse_args()

    # 環境変数からOpenAI APIキーと組織キーを取得
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.organization = os.getenv("OPENAI_ORG_KEY")

    # 各種パラメータの初期設定
    iteration_round = args.iteration_round
    temp = args.temperature
    n = args.n
    batchsize = args.batchsize
    model = args.model
    metrics = args.criterion
    
    # データ関連のディレクトリ・ファイル名設定
    data_name_ = "generate_fed_dialog_heterogeneous_{}/".format(metrics[:3].lower())
    dir_name_ = data_name_
    sub_name = "batch"
    prompt_name = "{}_analyze_{}.txt".format(metrics[:3], sub_name)
    level = ['turn', 'dialog'][1]  # ここでは "dialog" を使用
    prompt_dir = "prompts/fed_{}/{}".format(level, prompt_name)
    prompt = open(prompt_dir).read()  # プロンプトファイルを読み込む
    data_name_ = "eval_data_new/" + data_name_
    further_dir = "{}_{}_{}_{}_{}_{}".format(metrics, batchsize, temp, n, sub_name, model)

    # 各イテレーション（評価ラウンド）毎の処理
    for data_idx in range(iteration_round):
        print("Round:{}".format(data_idx))
        data_name = data_name_ + "{}.json".format(data_idx)
        dir_name = dir_name_ + "{}.json".format(data_idx)
        print(data_name)
        print("{}".format(further_dir))
        
        # 入力データの読み込み
        with open(data_name, "r") as f:
            datas = json.load(f)
        
        # 出力先ディレクトリの設定と作成
        dir_name = "API_Completion/" + dir_name + "/" + further_dir
        os.makedirs(dir_name, exist_ok=True)
        os.makedirs(dir_name + "/raw_data", exist_ok=True)
        
        # 現在処理中のバッチインデックスを管理するファイル
        if not os.path.exists(dir_name + "/current.txt"):
            with open(dir_name + "/current.txt", "w") as f:
                f.write("0")
        with open(dir_name + "/current.txt", "r") as f:
            start = int(f.read())
        
        # API利用コストの管理
        if not os.path.exists(dir_name + "/cost.txt"):
            with open(dir_name + "/cost.txt", "w") as f:
                f.write("0 0")
        with open(dir_name + "/cost.txt", "r") as f:
            tem_text = f.read()
            cost_now = {'prompt': int(tem_text.split(" ")[0]), 'completion': int(tem_text.split(" ")[1])}

        Model = Get()
        # 各バッチごとの評価処理
        for idx in range(start, len(datas["context"]) // batchsize + 1):
            if idx * batchsize >= len(datas["context"]):
                continue
            # 最終バッチの場合、バッチサイズを調整
            in_batchsize = batchsize
            if idx == len(datas["context"]) // batchsize:
                in_batchsize = len(datas["context"]) % batchsize
            
            # 各サンプルのデータを辞書型でまとめる（"dialog" レベルの場合）
            if level == "dialog":
                tem = [{key: datas[key][idx * batchsize + i] for key in datas.keys()}
                       for i in range(in_batchsize)]
            
            # 複数サンプルの入力データ（会話履歴等）の組み立て
            s = ""
            for i in range(in_batchsize):
                s += "\n\nSample{}:\n\nConversation History:\n".format(i + 1)
                s += datas["context"][idx * batchsize + i]
                if level == "turn":
                    s += "\nResponse:\n"
                    s += datas["response"][idx * batchsize + i] + "\n\n"
            
            # プロンプトのテンプレートにバッチデータを埋め込む
            cur_prompt = prompt.replace('{{Data}}', s).replace('{{number}}', str(in_batchsize))
            for i in range(in_batchsize):
                tem[i]['prompt'] = cur_prompt
            
            # API呼び出し（例外発生時はリトライ）
            while True:
                try:
                    response, cost = Model.calc(cur_prompt, n=n, temp=temp, model=model)
                    print("Batch idx:", idx)
                    for key in cost.keys():
                        cost_now[key] += cost[key]
                    # バッチの結果をJSON形式で保存
                    with open(dir_name + "/raw_data/{}.json".format(idx), "w") as f:
                        cur_t = {"raw_data": tem, "response": response}
                        json.dump(cur_t, f)
                    # 現在のバッチインデックスとコストを更新
                    with open(dir_name + "/current.txt", "w") as f:
                        f.write(str(idx + 1))
                    with open(dir_name + "/cost.txt", "w") as f:
                        f.write("{} {}".format(cost_now['prompt'], cost_now['completion']))
                    break
                except Exception as e:
                    print("Error:", e)
                    print("Sleeping for 10s...")
                    time.sleep(10)
        
        # 各ラウンド終了後に新しい評価データを生成する
        generate_new_data(dir_name_, data_idx, further_dir, batchsize)
    print("Complete Generation!")