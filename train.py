"""Run five signer-independent tests over a fixed ten-gloss vocabulary."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sequence import ROOT, CACHE, CLASSES, SIGNERS, HELD, SignLSTM, load_archive, predict


def summaries(x):
    return np.concatenate([x.mean(1), x.std(1), x.min(1), x.max(1)], axis=1)


def scores(y, p):
    return {"accuracy": round(float(accuracy_score(y,p)), 4),
            "macro_f1": round(float(f1_score(y,p,average="macro",zero_division=0)), 4)}


def fit(x, y, train, val, seed):
    torch.manual_seed(seed)
    mean, scale = x[train].mean((0,1)), x[train].std((0,1)).clip(.001)
    xt, yt = torch.as_tensor((x-mean)/scale), torch.as_tensor(y)
    model = SignLSTM()
    optimiser = torch.optim.AdamW(model.parameters(), lr=.003)
    loss = nn.CrossEntropyLoss()
    best, best_weights, best_epoch = float("inf"), None, 0
    for epoch in range(1, 41):
        model.train()
        for batch in torch.as_tensor(train)[torch.randperm(len(train))].split(32):
            optimiser.zero_grad()
            loss(model(xt[batch]), yt[batch]).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimiser.step()
        model.eval()
        with torch.inference_mode():
            metric = loss(model(xt[val]), yt[val]).item()
        if metric < best - .001:
            best, best_weights, best_epoch = metric, deepcopy(model.state_dict()), epoch
        if epoch - best_epoch >= 8:
            break
    model.load_state_dict(best_weights)
    return model.eval(), mean, scale, best_epoch


def run(archive):
    torch.set_num_threads(2)
    x, y, groups, _ = load_archive(archive)
    bag = summaries(x)
    folds, all_test, all_lstm, all_baseline = [], [], [], []
    for i, test_person in enumerate(SIGNERS):
        validation_person = SIGNERS[(i+1)%len(SIGNERS)]
        train = np.flatnonzero((groups!=test_person)&(groups!=validation_person))
        val = np.flatnonzero(groups==validation_person)
        test = np.flatnonzero(groups==test_person)
        model, mean, scale, selected = fit(x,y,train,val,821+i)
        lstm = predict(model,x[test],mean,scale).argmax(1)
        baseline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=800)).fit(bag[train],y[train]).predict(bag[test])
        fold = {"held_out": "ABCDE"[i], "validation": "ABCDE"[(i+1)%5],
                "train": len(train), "validation_count": len(val), "test": len(test), "selected_epoch": selected,
                "lstm": scores(y[test],lstm), "baseline": scores(y[test],baseline)}
        folds.append(fold)
        all_test.extend(y[test]); all_lstm.extend(lstm); all_baseline.extend(baseline)
        if test_person == HELD:
            torch.save({"state": model.state_dict(), "mean": torch.tensor(mean),
                        "scale": torch.tensor(scale)}, ROOT/"model.pt")
        print(fold,flush=True)
    report = {"classes": list(CLASSES), "total":len(y), "demo_fold":"C", "folds":folds,
              "lstm":scores(all_test,all_lstm), "baseline":scores(all_test,all_baseline),
              "confusion":confusion_matrix(all_test,all_lstm,labels=range(len(CLASSES))).tolist()}
    (ROOT/"report.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=CACHE)
    run(parser.parse_args().archive)
