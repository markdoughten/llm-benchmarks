# custom libraries
from lib import helpers
import ollama
import pandas as pd
import matplotlib.pyplot as plt

def run(models, question, gpu, graph, loops=1):
   
    # ask the models the prompts 
    performances = []
    while loops != 0: 
        for model in models:
            performances.append(run_model(model, gpu, question))
        loops -=1  
    
    # calculate metrics 
    metrics = calculate(performances)
    metrics = pd.DataFrame(metrics)
   
    if graph: 
        # plot time and count 
        plot(metrics, 'Seconds', ['total_duration', 'load_duration', 'prompt_eval_duration', 'eval_duration'], gpu)
        plot(metrics, 'Count', ['prompt_eval_count', 'eval_count'], gpu)
        plot(metrics, 'Token per Second', ['prompt_eval_per_second',  'eval_per_second', 'total_per_second'], gpu)
    
    return metrics

def run_model(model, gpu, prompt):

    if gpu:
        return ollama.generate(model=model, prompt=prompt, options={'main_gpu':0, 'num_gpu':33})
    else:
        return ollama.generate(model=model, prompt=prompt, options={'main_gpu':0, 'num_gpu':0})

def calculate(performances):
    nanoseconds = 10**9
    for model in performances:
        keys = ['load_duration', 'total_duration', 'prompt_eval_duration', 'eval_duration']
        # convert to seconds
        for key in keys:
            model[key] = model[key]/nanoseconds
        
        # calculate time per count
        model['prompt_eval_per_second'] = model['prompt_eval_duration']/model['prompt_eval_count']
        model['eval_per_second'] = model['eval_duration']/model['eval_count']
        model['total_per_second'] = model['total_duration']/(model['eval_count']+model['prompt_eval_count'])

    return performances
    
def plot(metrics, units, keys, gpu):
   
    keys.append('model')
    metrics = metrics[keys] 

    fig, ax = plt.subplots(figsize=(8, 8), dpi=96)   
    metrics.set_index('model').groupby(level=0).median().plot(kind='bar', ax=ax, rot=0)
    
    if gpu:
        plt.title('GPU')
    else: 
        plt.title('CPU')    
     
    plt.xlabel('Model')
    plt.ylabel(units)
    plt.tight_layout()
    plt.show()
        
    return

def save(df, gpu):
    
    if gpu:
        prefix = 'gpu'
    else:
        prefix = 'cpu'
    
    return df.to_csv(f'../data/{prefix}_metrics.csv', index=False)

def get_model():
    models = []
    for model in ollama.list()['models']:
        models.append(model['name']))
    return models

if __name__ == '__main__':
    
    gpu = True
    graph = True
    models = get_models()
    question = "Where is Rutger's University?"
   
    # run on the gpu 
    metrics = run(models, question, gpu, graph, loops=10)
    save(metrics, gpu) 
