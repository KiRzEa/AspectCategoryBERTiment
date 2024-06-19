from sklearn.metrics import *
import pandas as pd

def evaluation_scores(y_test, y_pred,time_training,inference_time):
    accuracy_test = round(accuracy_score(y_test, y_pred)*100,4)
    balance_accuracy = round(balanced_accuracy_score(y_test, y_pred)*100,4)
    f1_weighted_test = round(f1_score(y_test, y_pred, average='weighted')*100,4)
    f1_micro_test = round(f1_score(y_test, y_pred, average='micro')*100,4)
    f1_macro_test = round(f1_score(y_test, y_pred, average='macro')*100,4)

    scores = "Accuracy: " + str(accuracy_test) + "\nBalance Accuracy: " + str(balance_accuracy) \
                + "\nWeighted F1-score: " + str(f1_weighted_test) \
                + "\nMacro F1-score: " + str(f1_macro_test) \
                + "\nMicro F1-score: " + str(f1_micro_test) \
                + "\nTraining time: " + str(time_training) \
                + "\nInference time: " + str(inference_time)
    
    scores += "\n" + str(classification_report(y_test, y_pred)) + "\n"
    print("===============\n")
    print(scores)
    return scores

def export_score_to_file(scores, model_id, domain):
    text_score = "Model: " + model_id +"\n Domain: " + domain + "\n" + scores + "\n\n"
    score_output_path = "scores/"+ "scores_" + str(model_id.replace("/", "_")) + ".txt"
    with open(score_output_path, 'a') as file:
        file.write(text_score)
    print("Save file done: ", score_output_path)
    

def save_prediction_to_file(x_test, y_true, y_pred, domain, model_id):
    path_output = "prediction/"+ str(domain) + "_" + str(model_id.replace("/", "_")) + ".csv"
    df = pd.DataFrame(list(zip(x_test, y_true, y_pred)), columns =['review', 'y_true', 'y_pred'])
    df.to_csv(path_output, index=False)
    
    return df
