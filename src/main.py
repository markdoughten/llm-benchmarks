# custom libraries
from lib import questions
import ollama
import pandas as pd
import matplotlib.pyplot as plt
import random

def run(models, questions, gpu=True, write=False, graph=True, asks=1, display=['bar']):
    
    # ask the models the prompts 
    performances = []
    num_asks = asks
    for question in questions:
        while asks != 0: 
            for model in models:
                performances.append(run_model(model, gpu, question))
            asks -=1  
        
    # calculate metrics 
    metrics = calculate(performances)
    metrics = pd.DataFrame(metrics)
   
    if graph: 
        for kind in display: 
            # plot time and count 
            plot(metrics, 'Seconds', ['total_duration', 'load_duration', 'prompt_eval_duration', 'eval_duration'], gpu, kind, write, num_asks)
            plot(metrics, 'Count', ['prompt_eval_count', 'eval_count'], gpu, kind, write, num_asks)
            plot(metrics, 'Token per Second', ['prompt_eval_per_second',  'eval_per_second', 'total_per_second'], gpu, kind, write, num_asks)
     
    return metrics

def run_model(model, gpu, prompt):

    output = {'prompt': prompt, 'gpu': gpu}

    if gpu:
        data = ollama.generate(model=model, prompt=prompt, options={'main_gpu':0, 'num_gpu':33})
    else:
        data = ollama.generate(model=model, prompt=prompt, options={'main_gpu':0, 'num_gpu':0})

    return merge_two_dicts(output, data)

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
        model['token_per_second'] = (model['eval_count']+model['prompt_eval_count'])/model['total_duration']

    return performances
    
def plot(metrics, units, keys, gpu, kind, write, asks):
   
    keys.append('model')
    info = f"Q:{len(metrics['prompt'].unique())}, A:{asks}"
    metrics = metrics[keys] 

    fig, ax = plt.subplots(figsize=(8, 8), dpi=96)

    if kind == 'box':
         metrics.set_index('model').plot(kind=kind, ax=ax, rot=0)
    else:
        metrics.set_index('model').groupby(level=0).median().plot(kind=kind, ax=ax, rot=0)
    
    if gpu:
        plt.title(f'GPU: {info}')
    else: 
        plt.title(f'CPU: {info}')    
     
    plt.ylabel(units)
    plt.tight_layout()

    if write:
       save(chart=True, gpu=gpu, plt=plt)

    #plt.show()
    
    return

def save(df=None, gpu=None, chart=False, plt=None):
    
    if gpu:
        prefix = 'gpu'
    else:
        prefix = 'cpu'
    
    if chart:
        plt.savefig(f'../data/charts/{prefix}/{random.randint(1, 100)}.png')
    else:
        df.to_csv(f'../data/metrics.csv', index=False)
    
    return 

def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z

def get_models():
    models = []
    for model in ollama.list()['models']:
        models.append(model['name'])
    return models

if __name__ == '__main__':
    
    random_select = True
    write = True
    gpu = True
    num_questions = 1
    num_asks = 10
    models = get_models()
    questions = questions.get_questions()
    both = True

    if num_questions > len(questions):
        print('n is too big, only {len(questions)}')
        exit()

    if random_select:
        questions = random.sample(questions, k=num_questions)
   
    metrics = run(models, questions, gpu, write, graph=True, asks=num_asks, display=['box', 'bar'])
    
    # run on the opposite 
    if both:
        gpu = not gpu
        metrics = pd.concat([run(models, questions, gpu, write, graph=True, asks=num_asks, display=['box', 'bar']), metrics])
   
    if write:
        save(metrics, gpu) 

