# custom libraries
from lib import helpers
import ollama
import pandas as pd
import matplotlib.pyplot as plt

def run(models, ask, gpu):
   
    # ask the models the prompts 
    performances = []
    for model in models:
        if gpu:
            performances.append(ollama.generate(model=model, prompt=ask, options={'main_gpu':0, 'num_gpu':33}))
        else:
            performances.append(ollama.generate(model=model, prompt=ask, options={'main_gpu':0, 'num_gpu':0}))
    
    # calculate metrics 
    metrics = calculate(performances)
    metrics = pd.DataFrame(metrics)
    
    # plot time and count 
    plot(metrics, 'Seconds', ['total_duration', 'load_duration', 'prompt_eval_duration', 'eval_duration'], gpu)
    plot(metrics, 'Count', ['prompt_eval_count', 'eval_count'], gpu)
    plot(metrics, 'Token per Second', ['prompt_eval_per_second',  'eval_per_second', 'total_per_second'], gpu)
    
    return metrics

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
    metrics.set_index('model').plot(kind='bar', ax=ax, rot=0)
    
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

if __name__ == '__main__':
    
    models = ['llama2:7b-chat-q4_0', 'llama2:7b-chat-q5_0', 'llama2:7b-chat-q8_0']
    gpu = False
    ask = "Where is Rutgers University?"
   
    # run on the gpu 
    metrics = run(models, ask, gpu)
    save(metrics, gpu) 
