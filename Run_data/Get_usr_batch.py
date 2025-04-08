import glob
import json
import random
import openai
import os
import time
import re
import argparse

# OpenAI API問い合わせクラス
class Get:
    def __init__(self):
        self.prompt = ""
        
    def calc(self, query, temp=0, n=1, model='gpt-4-1106-preview'):
        # 指定のプロンプトを使い、OpenAIのChatCompletionを呼び出す
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "system", "content": query}],
            temperature=temp,
            top_p=0.9,
            n=n
        )
        res = [tem["message"]["content"] for tem in response["choices"]]
        return res, {
            'prompt': response['usage']['prompt_tokens'],
            'completion': response['usage']['completion_tokens']
        }

# レスポンスから数値スコア群を抽出する関数
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

# 生データから新たな評価用データを生成し、JSONファイルに出力する関数
def generate_new_data(dir_name, data_idx, further_dir, batchsize):
    generate_name = "API_Completion/" + dir_name + "{}.json/{}/raw_data/*".format(data_idx, further_dir)
    dirs = glob.glob(generate_name)
    raw_datas = []
    for file in dirs:
        with open(file, "r") as f:
            l = json.load(f)
        raw_datas.append(l)
    datas = []
    for tem in raw_datas:
        score = get_score(tem['response'][0], num=batchsize)
        for i in range(len(tem['raw_data'])):
            tem['raw_data'][i]['modelscore'] = score[i]
            datas.append(tem['raw_data'][i])
    # スコア順にソート
    datas.sort(key=lambda x: x['modelscore'], reverse=True)
    new_datas = []
    bottle = len(datas) // batchsize
    for i in range(bottle):
        cur_batch = []
        for j in range(batchsize):
            cur_batch.append(datas[bottle * j + i])
        random.shuffle(cur_batch)
        new_datas += cur_batch
    new_batch = {}
    for key in new_datas[0].keys():
        new_batch[key] = [t[key] for t in new_datas]
    with open("eval_data_new/" + dir_name + "{}.json".format(data_idx + 1), "w") as f:
        json.dump(new_batch, f)

# メイン処理開始
if __name__ == '__main__':
    #コマンドライン引数の指定
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--iteration_round', type=int, default=5)
    argparser.add_argument('--temperature', type=float, default=0.2)
    argparser.add_argument('--n', type=int, default=1)
    argparser.add_argument("--batchsize", type=int, default=10)
    argparser.add_argument('--model', type=str, default="gpt-4-1106-preview")
    argparser.add_argument('--criterion', type=str, default='engagingness')
    args = argparser.parse_args()

    # 環境変数からAPIキーおよび組織キーを取得
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.organization = os.getenv("OPENAI_ORG_KEY")

    iteration_round = args.iteration_round
    temp = args.temperature
    n = args.n
    batchsize = args.batchsize
    model = args.model
    metrics = args.criterion
    
    data_name_ = "generate_usr_topical_heterogeneous_{}/".format(metrics[:3])
    dir_name_ = data_name_
    sub_name = "batch"
    prompt_name = "{}_analyze_{}.txt".format(metrics[:3], sub_name)
    prompt_dir = "prompts/topical_chat/{}".format(prompt_name)
    prompt = open(prompt_dir).read()
    data_name_ = "eval_data_new/" + data_name_
    further_dir = "{}_{}_{}_{}_{}_{}".format(metrics, batchsize, temp, n, sub_name, model)

    # 各イテレーションごとに評価データを読み込み、API呼び出しや結果の保存を行う
    for data_idx in range(iteration_round):
        data_name = data_name_ + "{}.json".format(data_idx)
        dir_name = dir_name_ + "{}.json".format(data_idx)
        with open(data_name, "r") as f:
            datas = json.load(f)
        dir_name = "API_Completion/" + dir_name + "/" + further_dir
        os.makedirs(dir_name, exist_ok=True)
        os.makedirs(dir_name + "/raw_data", exist_ok=True)
        if not os.path.exists(dir_name + "/current.txt"):
            with open(dir_name + "/current.txt", "w") as f:
                f.write("0")
        with open(dir_name + "/current.txt", "r") as f:
            start = int(f.read())
        if not os.path.exists(dir_name + "/cost.txt"):
            with open(dir_name + "/cost.txt", "w") as f:
                f.write("0 0")
        with open(dir_name + "/cost.txt", "r") as f:
            tem_text = f.read()
            cost_now = {'prompt': int(tem_text.split(" ")[0]), 'completion': int(tem_text.split(" ")[1])}

        # get_context関数：対話文を整形して連結する
        def get_context(tem):
            start_s = "Speaker B: " if len(tem) % 2 else "Speaker A: "
            s = ""
            for t in tem:
                s += start_s + t.strip() + "\n"
                start_s = "Speaker A: " if start_s == "Speaker B: " else "Speaker B: "
            return s

        Model = Get()
        for idx in range(start, len(datas["context"]) // batchsize):
            # 各バッチのデータを辞書形式でまとめる
            tem = [{key: datas[key][idx * batchsize + i] for key in datas.keys()} for i in range(batchsize)]
            # 各サンプルのソーステキストを整形
            source = [get_context(datas["source"][idx * batchsize + i].split("\n")[:-2]) for i in range(batchsize)]
            s = ""
            for i in range(batchsize):
                # サンプル毎の会話履歴とシステム出力を連結
                s += "Sample{}:\n\nConversation History:\n".format(i + (5 if "icl" in sub_name else i + 1))
                s += source[i]
                s += "\nResponse:\n"
                s += "Speaker A: " + datas["system_output"][idx * batchsize + i] + "\n\n"
            # プロンプトテンプレートにデータを埋め込む
            cur_prompt = prompt.replace('{{Data}}', s).replace('{{number}}', str(batchsize))
            for i in range(batchsize):
                tem[i]['prompt'] = cur_prompt
            while True:
                try:
                    response, cost = Model.calc(cur_prompt, n=n, temp=temp, model=model)
                    print("Batch idx:", idx)
                    for key in cost.keys():
                        cost_now[key] += cost[key]
                    with open(dir_name + "/raw_data/{}.json".format(idx), "w") as f:
                        cur_t = {"raw_data": tem, "response": response}
                        json.dump(cur_t, f)
                    with open(dir_name + "/current.txt", "w") as f:
                        f.write(str(idx + 1))
                    with open(dir_name + "/cost.txt", "w") as f:
                        f.write("{} {}".format(cost_now['prompt'], cost_now['completion']))
                    break
                except Exception as e:
                    print("Error:", e)
                    print("Sleeping for 10s...")
                    time.sleep(10)
        generate_new_data(dir_name_, data_idx, further_dir, batchsize)
    print("Complete Generation!")