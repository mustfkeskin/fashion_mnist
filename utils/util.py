import os
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly import tools

from sklearn.metrics import precision_recall_fscore_support, confusion_matrix


def create_trace(x,y,ylabel,color):
        trace = go.Scatter(
                x = x,
                y = y,
                name = ylabel,
                marker=dict(color = color),
                mode = "markers+lines",
                text = x
            )
        return trace
    
def plot_accuracy_and_loss(history, model_name):
    hist     = history.history
    acc      = hist['acc']
    val_acc  = hist['val_acc']
    loss     = hist['loss']
    val_loss = hist['val_loss']
    epochs = list(range(1,len(acc)+1))
    
    trace_ta = create_trace(epochs,acc,"Training accuracy", "Green")
    trace_va = create_trace(epochs,val_acc,"Validation accuracy", "Red")
    trace_tl = create_trace(epochs,loss,"Training loss", "Blue")
    trace_vl = create_trace(epochs,val_loss,"Validation loss", "Magenta")
   
    fig = tools.make_subplots(rows=1,cols=2, subplot_titles=('Training and validation accuracy',
                                                             'Training and validation loss'))
    fig.append_trace(trace_ta,1,1)
    fig.append_trace(trace_va,1,1)
    fig.append_trace(trace_tl,1,2)
    fig.append_trace(trace_vl,1,2)
    fig['layout']['xaxis'].update(title = 'Epoch')
    fig['layout']['xaxis2'].update(title = 'Epoch')
    fig['layout']['yaxis'].update(title = 'Accuracy', range=[0,1])
    fig['layout']['yaxis2'].update(title = 'Loss', range=[0,1])

    layout = go.Layout(title = model_name)
    fig = go.Figure(data=fig, layout=layout)
    iplot(fig, filename=model_name + ".png")

def save_model(model, model_name):
    save_dir = "../Models/"
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)



def accuracyMetrics2df(prfs, target_names):
    df = pd.DataFrame()
    for p,r,f,s,label in zip(prfs[0], prfs[1], prfs[2], prfs[3], target_names):
        rowdata={}
        rowdata['precision'] = p
        rowdata['recall']    = r
        rowdata['f1-score']  = f
        rowdata['support']   = s
        df = df.append(pd.DataFrame.from_dict({label:rowdata}, orient='index'))   
    return df[['precision', 'recall', 'f1-score', 'support']]


def get_pred_and_metrics(model, X_test, y_true, target_names):
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred,axis=1)

    prfs = precision_recall_fscore_support(y_true, y_pred)
    df   = accuracyMetrics2df(prfs, target_names = target_names.values())
    return y_pred, df

def majorityVoting(pred_df, y_true, target_names):
    y_pred = np.asarray([np.argmax(np.bincount(pred_df.loc[row,:])) for row in range(pred_df.shape[0])])

    prfs = precision_recall_fscore_support(y_true, y_pred)
    df   = accuracyMetrics2df(prfs, target_names = target_names.values())
    return df

def plot_confussion_matrix(y_true, y_pred, target_names, model_name):
    z = confusion_matrix(y_true, y_pred) 
    x = list(target_names.values())
    y = list(target_names.values())

    fig = ff.create_annotated_heatmap(z, x=x, y=y)
    layout = go.Layout(title = model_name)

    fig = go.Figure(data=fig, layout=layout)
    iplot(fig, filename=model_name + ".png")