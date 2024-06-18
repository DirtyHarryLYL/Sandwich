import json
import glob
import pickle
import numpy as np
import torch
import clip
from PIL import Image
import tqdm
import requests

import os
os.chdir("..")

def organize_verbs():
    ## target node
    mapping_node_index = pickle.load(open("./Data/mapping_node_index/704_to_290.pkl", "rb"))
    verbnet_topology = pickle.load(open(os.path.join("Data", "verbnet_topology_898.pkl"), "rb"))
    objects = np.array(verbnet_topology["objects"]) # name of 898 hierarchical nodes
    target_nodes = objects[mapping_node_index]
    NODE2ID = {node:i for i, node in enumerate(target_nodes)}
    Node2Info = json.load(open("./Data/Node2Info_format.json", "r"))
    
    Node2Verbs = {}
    for node in target_nodes:
        Node2Verbs[node] = {}
        infos = Node2Info[node]
        for i, item in enumerate(infos):
            if item.startswith("An example sentence: "):
                continue
            else:
                if i > 4:
                    verb_member = item[:item.find('.')]
                    Node2Verbs[node][verb_member] = 1
        
    cand_verbs = []
    info_verbs = []
    for node in Node2Verbs.keys():
        for verb in Node2Verbs[node].keys():
            if Node2Verbs[node][verb] == 1:
                flag = False
                for info in Node2Info[node]:
                    if info.split('.')[0] == verb:
                        info = info[:info.find("The example sentences are")] + info[info.find("The lexname is:"):]
                        info_verbs.append(info)
                        flag = True
                if flag:
                    cand_verbs.append((node, verb))
                # else:
                #     print(node, verb)
    print("cand verbs total: ", len(cand_verbs))
    return cand_verbs, info_verbs


def organize_actions(dataset_name):

    collected_activities = []
    path = "./semantic_alignment/dataset_labels/%s.txt"%dataset_name
    with open(path, "r") as f:
        for line in f.readlines():
            action = line.strip('\n').strip()
            collected_activities.append((dataset_name, action))
    collected_activities = np.array(collected_activities)
    return collected_activities


def align_with_clip(dataset_name):
    
    # device = "cpu"
    device="cuda:6"
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    def get_clip_text_embed(text_list):
        text = clip.tokenize(text_list).to(device)
        with torch.no_grad():
            text_features = clip_model.encode_text(text)
        return text_features
    
    cand_labels = organize_actions(dataset_name)
    # print(dataset_name, len(cand_labels))
    # return
    cand_verbs, info_verbs = organize_verbs()
    
    cand_labels_single = [c[1] for c in cand_labels]
    cand_verbs_single = [c[1] for c in cand_verbs]
    feature_labels = get_clip_text_embed(cand_labels_single)
    feature_verbs = get_clip_text_embed(info_verbs)
    map_label2verb = (feature_labels / feature_labels.norm(dim=1, keepdim=True)) @ (feature_verbs / feature_verbs.norm(dim=1, keepdim=True)).t()
    print(dataset_name, map_label2verb.shape)
    os.makedirs("./semantic_alignment/dataset_labels_align_result/%s"%dataset_name, exist_ok=True)
    pickle.dump({
        "map_label2verb": map_label2verb,
        "node_verb_list": cand_verbs,
        "verb_text_list": info_verbs,
        "label_list": cand_labels,
        }, open("./semantic_alignment/dataset_labels_align_result/%s/map_label2verb_clip.pkl"%dataset_name, "wb"))


def chatgpt_demo_new(content):
    headers = {
        'content-type': 'application/json',
        "Authorization": "Bearer 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",
    }
    url = 'https://api.openai-proxy.com/v1/chat/completions'
    # url = 'https://chatgpt1.nextweb.fun/api/proxy/v1/chat/completions'
    data = {"model": "gpt-3.5-turbo", "messages": content,"temperature": 0.7}
    response = requests.post(url, headers=headers, json=data)
    return json.loads(response.content)["choices"][0]["message"]


def align_with_gpt(dataset_name):

    data = pickle.load(open("./semantic_alignment/dataset_labels_align_result/%s/map_label2verb_clip.pkl"%dataset_name, "rb"))
    map_label2verb = data["map_label2verb"].cpu().numpy()
    node_verb_list = data["node_verb_list"]
    verb_text_list = data["verb_text_list"]
    label_list = data["label_list"]
    # for i in range(len(label_list)):
    #     print(i, (map_label2verb[i]>0.8).sum(), (map_label2verb[i]>0.7).sum(), (map_label2verb[i]>0.6).sum())

    for i in tqdm.trange(len(label_list)):
    # for i in tqdm.tqdm([446]):
        candidate_verbs_idx = np.argsort(-map_label2verb[i])[:50]
        candidate_verbs = np.array([node_verb_list[idx][1] for idx in candidate_verbs_idx])
        txts  = "\n".join([str(idx) + ". " + verb_text_list[idx] for idx in candidate_verbs_idx])
        prompt = """Select 10 verbs related to the activity classes "%s".
There are 50 candidate verbs: 
%s

Output Format Example:
%d. %s.
%d. %s.
Strictly follow the output format above. Do not output analysis. 
"""%(label_list[i][1], txts,
    candidate_verbs_idx[0], candidate_verbs[0],
    candidate_verbs_idx[1], candidate_verbs[1],
    )
        # print(prompt)
        if os.path.exists("./semantic_alignment/dataset_labels_align_result/%s/answer_%d.txt"%(dataset_name, i)):
            continue
        with open("./semantic_alignment/dataset_labels_align_result/%s/prompt_%d.txt"%(dataset_name, i), "w") as f:
            f.write(prompt)
        INPUT = [
            {"role": "user", "content": prompt},
        ]
        ans = chatgpt_demo_new(INPUT)['content']
        with open("./semantic_alignment/dataset_labels_align_result/%s/answer_%d.txt"%(dataset_name, i), "w") as f:
            f.write(ans)
        # print(ans)
        # break


def post_process(dataset_name):

    data = pickle.load(open("./semantic_alignment/dataset_labels_align_result/%s/map_label2verb_clip.pkl"%dataset_name, "rb"))
    map_label2verb = data["map_label2verb"].cpu().numpy()
    node_verb_list = data["node_verb_list"]
    verb_list = np.array([item[1] for item in node_verb_list])
    verb_text_list = data["verb_text_list"]
    label_list = data["label_list"]
    Label2Verb = {}
    missing_list = []
    
    for i in range(len(label_list)):
        try:
            with open("./semantic_alignment/dataset_labels_align_result/%s/answer_%d.txt"%(dataset_name, i), "r") as f:
                answer = f.read()
            selected_verb_idx = []
            for line in answer.split('\n'):
                selected_verb_idx.append(int(line.split('.')[0]))
            # print(selected_verb_idx)
            related_verbs = []
            for idx in selected_verb_idx:
                related_verbs.append(node_verb_list[idx])
            # print(i, label_list[i], related_verbs)
            Label2Verb[(i, label_list[i][1])] = related_verbs
        except:
            # try:
            with open("./semantic_alignment/dataset_labels_align_result/%s/answer_%d.txt"%(dataset_name, i), "r") as f:
                answer = f.read()
            related_verbs = []
            for line in answer.split('\n'):
            # for idx in selected_verb_idx:
                verb = line.strip().strip('.')
                idxs = np.where(verb_list == verb)[0]
                if len(idxs) > 0:
                    idx = np.where(verb_list == verb)[0][0]
                    related_verbs.append(node_verb_list[idx])
            # print(i, label_list[i], related_verbs)
            if len(related_verbs) > 3:
                Label2Verb[(i, label_list[i][1])] = related_verbs
            else:
                missing_list.append(i)
            # except:
            #     missing_list.append(i)

    if len(Label2Verb) == len(label_list):
        pickle.dump({
            "Label2Verb": Label2Verb,
            "node_verb_list": node_verb_list,
            "verb_text_list": verb_text_list,
            "label_list": label_list,
            }, open("./semantic_alignment/dataset_labels_align_result/%s/map_label2verb_clip+gpt.pkl"%dataset_name, "wb"))
    else:
        print(len(Label2Verb), len(label_list))
        print("missing_list: ", missing_list)

if __name__ == "__main__":
    
    ## align
    '''
    for path in glob.glob("semantic_alignment/dataset_labels/*.txt"):
        dataset_name = path.split('/')[-1][:-4]
        print(dataset_name)

        ## align with clip
        # if os.path.exists("./semantic_alignment/dataset_labels_align_result/%s/map_label2verb_clip.pkl"%dataset_name):
        #     continue
        # align_with_clip(dataset_name)

        ## align with gpt
        # if os.path.exists("./semantic_alignment/dataset_labels_align_result/%s/map_label2verb_clip+gpt.pkl"%dataset_name):
        #     continue
        # align_with_gpt(dataset_name)

        ## post process
        post_process(dataset_name)
        print('------------')
    '''
    
    ## adjust a few failure cases
    # dataset_name = "Sports-1M"
    # align_with_gpt(dataset_name)
    # post_process(dataset_name)

    ## check
    for path in glob.glob("semantic_alignment/dataset_labels/*.txt"):
        dataset_name = path.split('/')[-1][:-4]
        print(dataset_name)
        assert os.path.exists("./semantic_alignment/dataset_labels_align_result/%s/map_label2verb_clip+gpt.pkl"%dataset_name)