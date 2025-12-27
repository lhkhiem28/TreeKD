import pandas as pd

def get_scores_generation(eval_outputs, tasktype, checkpoint_path):
    df = pd.concat([pd.DataFrame(output) for output in eval_outputs])
    df.to_csv(checkpoint_path.replace(".pth", ".csv"))

    if tasktype == "regression":
        from sklearn.metrics import mean_absolute_error
        from scipy.stats import spearmanr

        labels, preds = [], []
        for label, pred in zip(df['label'].values.tolist(), df['pred'].values.tolist()):
            try:
                label, pred = float(label), float(pred)
                if abs(pred) < 1e4:
                    labels.append(label), preds.append(pred)
            except:
                pass
        return [mean_absolute_error(labels, preds), spearmanr(labels, preds)[0], 100*len(preds)/len(df)]
    elif tasktype == "classification":
        from sklearn.metrics import roc_auc_score, average_precision_score

        labels, preds = [], []
        for label, pred, token_scores in zip(df['label'].values.tolist(), df['pred'].values.tolist(), df['token_scores'].values.tolist()):
            try:
                if label in ("Yes", "No") and pred in ("Yes", "No"):
                    label = 1 if label == "Yes" else 0
                    if pred == "Yes":
                        pred_score = next(score for token, score in token_scores if token == "Yes")
                        labels.append(label), preds.append(pred_score)
                    if pred == "No":
                        pred_score = next(score for token, score in token_scores if token == "No")
                        labels.append(label), preds.append(1 - pred_score)
                else:
                    pass
            except:
                pass
        return [roc_auc_score(labels, preds), average_precision_score(labels, preds), 100*len(preds)/len(df)]

eval_funcs = {
    'generation': get_scores_generation,
}