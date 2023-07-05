from typing import List, Dict, Optional, Tuple, Union
import itertools
import os
import os.path as osp
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
sns.set()

import rich
from marlpo.utils.debug import printPanel, print
rich.get_console().width += 30

# 设置需要的列：
# x = 'num_env_steps_trained'
x = 'timesteps_total'
succ = 'SuccessRate'
crash = 'CrashRate'
out = 'OutRate'
maxstep = 'MaxStepRate'
# reward = 'Reward' #TODO
all_col = [x, succ, crash, out, maxstep]
succ_col = [x, succ]
crash_col = [x, crash]
out_col = [x, out]
maxstep_col = [x, maxstep]

all_metrics = {
    'succ': succ_col, 
    'out': out_col, 
    'crash': crash_col, 
    'maxstep': maxstep_col,
}
# succ_rew_col = [x, succ, rew]

# 用于匹配目录名称中种子数的正则pattern
seed_pattern = r'start_seed=(\d*)'

def get_param_patthern(param_space: Dict[str, List[str]], verbose=False) -> Dict:
    ''' 根据一个包含参数名和相应参数值的参数空间字典
        自动从一个字典中生成所有参数空间的笛卡尔积
        并生成对应的正则表达式
        用于匹配实验目录名
    args:
        param_space: {
            param_name_0: [[label, ...], [param_value, ...]],
            param_name_1: [param_value, ...], NOTE: 参数空间也可以不带标签
            ...
        } NOTE: 生成的正则表达式内的各项也是按照字典键的顺序

    returns:
        一个字典 包含参数空间的键值对的排列 以其对应的标签 (plt绘图用) 为键 如:  {
            param_0_label & param_1_lable ... : 'param_name_0=param_0_value_0,param_name_1=param_1_value_0',
            ...
        }
    '''
    msg = {}
    msg['input params'] = f'{str(list(param_space.keys()))}'

    res = {}
    all_param_list = []
    all_label_list = []
    for key, space in param_space.items():
        one_param_list = []
        one_label_list = []
        # 如果有标签简写
        if isinstance(space[0], list) and len(space) == 2:
            labels = space[0]
            values = space[1]
            assert len(labels) == len(values) # 保证标签数与参数值数量一致
            all_label_list.append(labels)
            for v in values:
                one_param_list.append(f'{key}={v}')
        # 如果没有标签简写
        else:
            p_space = space
            for p in p_space:
                one_label_list.append(f'{key}={p}')
                one_param_list.append(f'{key}={p}')
            all_label_list.append(one_label_list)
        all_param_list.append(one_param_list)

    all_param_product = list(itertools.product(*all_param_list)) # list of tuple, a tuple is a product
    all_label_product = list(itertools.product(*all_label_list)) # list of tuple, a tuple is a product
   
    for l, p in zip(all_label_product, all_param_product):
        l = [x for x in list(l) if x != '']
        key = ' & '.join(l)
        pattern = r'(.*?)'.join(p)
        res[key] = pattern

    product_helper_str = 'x'.join([str(len(pl)) for pl in all_param_list])
    msg['param combinations'] = f'total {product_helper_str}={len(res)} combinations'
    msg['==']= '-'
    msg['label'] = 're pattern:'
    msg['-']= '-'
    for i, k in enumerate(res):
        msg[f'(label {i+1}) ' + k] = res[k]
    
    if verbose:
        printPanel(msg, title=f'generating total {len(param_space)} param spaces')

    return res


def read_csv_in_dir(exp_dir, param_pattern, columns: list, verbose=False):
    ''' 给定一个实验目录和一个正则表达式匹配式, 筛序出该目录下所有符合该表达式的目录
        并返回其中的progress.csv文件内所需要的列

        NOTE: 自动还会聚集同一个param_pattern的不同种子
        允许输入的pattern也是匹配种子的pattern, 结果不变
    '''
    trial_dirs = [entry for entry in os.listdir(exp_dir) if osp.isdir(osp.join(exp_dir, entry))]
    csv_list = []
    all_seeds = []

    seed_pattern = r'start_seed=(\d*)'

    if verbose:
        print(f'Reading csv in {exp_dir}...')
        trial_dirs = tqdm(trial_dirs)
    for d in trial_dirs:
        if re.search(param_pattern, d):
            match = re.search(seed_pattern, d)
            if match: 
                seed = match.group()[len('start_seed='):]
                all_seeds.append(seed)
            file_path = osp.join(exp_dir, d, 'progress.csv')
            progress = pd.read_csv(file_path)
            data = progress[columns]
            csv_list.append(data)

            if verbose:
                print('found param pattern:', param_pattern)
    
    if verbose:
        print(f'Finished reading {len(all_seeds)} seed: ', ', '.join(all_seeds))
    return csv_list


def plot_mean_std(
    data, 
    x: str, 
    col: List[str], 
    title=None, 
    lable=None,
    xlabel=True,
    ylabel=True,
):
    ''' 输入一个data 包含以同一组数据为x轴的多个种子的多列数据 
        按相同x值聚集某列的不同种子的数据
        并绘制均值、标准差
    '''
    for c in col:
        if c != x:
            mean_values = data.groupby(x)[c].mean()
            std_values = data.groupby(x)[c].std()

            # 绘制均值和标准差曲线
            label = lable if lable else c
            plt.plot(mean_values.index, mean_values.values, label=lable)
            plt.fill_between(mean_values.index, mean_values - std_values, mean_values + std_values, alpha=0.2)

            # 设置图例和标签
            plt.title(title)
            plt.legend(title='', loc='best')
            if xlabel:
                plt.xlabel('Environmental Step')
            if ylabel:
                plt.ylabel('rate')


def plot_one_exp(
        exp_dir, 
        param_pattern=None, 
        col=succ_col, 
        title=None, 
        exp_label=None, 
        xlabel=True,
        ylabel=True,
        verbose=False
    ):
    ''' 简单从一个exp目录根据pattern筛选所有trial
        分别读取csv文件, 再拼接, 
        然后按x轴聚集不同种子绘制均值、标准差图
    '''
    if param_pattern == None:
        param_pattern = r'start_seed=(\d*)'
    df_list = read_csv_in_dir(exp_dir, param_pattern, col, verbose=verbose)
    data = pd.concat(df_list)
    plot_mean_std(data, x, col, title=title, lable=exp_label, xlabel=xlabel, ylabel=ylabel)


def compare_all_metrics(exp_dirs: Dict[str, Union[str, Tuple[str, str]]]):
    ''' 从给定的目录中绘制所有的指标, 绘制一个2X2的图形
        用于不同实验在各个指标下的对比
    args:
        exp_dirs: dict of exp_name -> (exp_dir, param_pattern)
            if no param_pattern provided, use seed_pattern
    '''
    fig = plt.figure(figsize=(12, 8))
    fig.subplots_adjust(hspace=0.35, wspace=0.2)
    i = 1
    for metric, col in all_metrics.items():
        plt.subplot(2, 2, i)
        for label, v in exp_dirs.items():
            if isinstance(v, str):
                exp_dir = v
                pattern = None
            elif isinstance(v, tuple):
                exp_dir, pattern = exp_dirs[label]
            else:
                raise NotImplementedError
            plot_one_exp(exp_dir=exp_dir, param_pattern=pattern, col=col, title=metric, exp_label=label)
        i += 1


def plot_all_metrics_for_params_in_one_exp_dir(
    exp_dir: str, 
    algo_name: str,
    param_pattern_dict: Dict[str, str],
    title: str = '',
    additional_plot_args: dict = None,
    # a_exp_dir: str = None,
):
    ''' 绘制一个目录中不同超参设置下的所有实验的所有指标, 绘制一个2X2的图形
        用于同一个目录下不同超参在各个指标下的对比
    Args:
        kwargs: additional plot args
            {
                'exp_name': {
                    'exp_dir', 'pattern', 'label', ...
                } x N
            }
    '''
    fig = plt.figure(figsize=(12, 8))
    fig.subplots_adjust(top=0.92, hspace=0.25, wspace=0.2)
    i = 1
    for metric, col in all_metrics.items():
        plt.subplot(2, 2, i)
        xlabel = i >= 3

        for label, pattern in param_pattern_dict.items():
            plot_one_exp(exp_dir=exp_dir, param_pattern=pattern, col=col, title=metric, exp_label=algo_name+' '+label, xlabel=xlabel, ylabel=False)

        if additional_plot_args:
            for exp_name, args in additional_plot_args.items():
                a_exp_dir = args['exp_dir']
                a_pattern = args.get('pattern', None)
                a_label = args.get('label', '')
                plot_one_exp(exp_dir=a_exp_dir, param_pattern=a_pattern, col=col, title=metric, exp_label=exp_name+' '+a_label, xlabel=xlabel, ylabel=False)

        i += 1

    plt.suptitle(title)
    




if __name__ == "__main__":

    exp_dir = 'exp_results/IPPO_Inter_8-30agents_(compact-state)'
    param_space = {
        # 'agents': [['30a-add-1-nei-state-navi'], ['30']],
        'agents': [['',], ['8',]],
        'num_neighbours': [['1-nei', '4-nei'], ['1', '4']],
    }
    param_pattern_dict = get_param_patthern(param_space, verbose=True) # {lable -> re pattern}

    # exp_dir_30a = 'exp_results/IPPO_Intersection_8seeds_30agents_repeat2'
    # add_plot_args = dict(
    #     IPPO=dict(
    #         exp_dir=exp_dir_30a,
    #         pattern=None, 
    #         label='30a', 
    # ))

    # plot_all_metrics_for_params_in_one_exp_dir(exp_dir, 'IPPO', param_pattern_dict, 'Compact Ego- and Nei- states with Lidar', add_plot_args)

    # plt.show()