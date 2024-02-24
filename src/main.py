# custom libraries
from lib import helpers
import ollama
import pandas as pd
import matplotlib.pyplot as plt

def run(models, ask):
    performances = []
    for model in models:
        performances.append(ollama.generate(model=model, prompt=ask, options={'main_gpu':0, 'num_gpu':16}))
    standard = convert_seconds(performances)
    metrics = pd.DataFrame(standard)
    
    # plot time and count 
    plot(metrics, 'Seconds', ['total_duration', 'load_duration', 'prompt_eval_duration', 'eval_duration'])
    plot(metrics, 'Count', ['prompt_eval_count', 'eval_count'])

    return metrics

def convert_seconds(performances):
    units = 10**9
    for model in performances:
        keys = ['load_duration', 'total_duration', 'prompt_eval_duration', 'eval_duration']
        for key in keys:
            model[key] = model[key]/units
    return performances
    
def plot(metrics, units, keys):
   
    keys.append('model')
    
    metrics = metrics[keys]    
    metrics.set_index('model').plot(kind='bar', figsize=(10, 6), rot=0)
    
    plt.xlabel('Model')
    plt.ylabel(units)
    plt.tight_layout()
    plt.show()
        
    return

if __name__ == '__main__':
    models = ['llama2:7b-chat-q4_0', 'llama2:7b-chat-q5_0', 'llama2:7b-chat-q8_0']
    gpu = False
    metrics = run(models, 'Why is the sky blue?')
    print(metrics)
    metrics.to_csv('../data/metrics.csv', index=False)

