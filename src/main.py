# custom libraries
from lib import questions
import ollama
import pandas as pd
import matplotlib.pyplot as plt
import random

def run(models, questions, gpu, graph, loops=1):
   
    # ask the models the prompts 
    performances = []
    for question in questions:
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

    output = {'prompt': prompt}

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
    
    gpu = True
    graph = True
    random_select = True
    write = True
    n = 1
    models = get_models()
    questions = questions.get_questions()

    if n > len(questions):
        print('n is too big, only {len(questions)}')
        exit()

    if random_select:
        questions = random.sample(questions, k=n)
   
    # run on the gpu 
    metrics = run(models, questions, gpu, graph, loops=1)
    
    print(metrics.to_string())
    if write:
        save(metrics, gpu) 

