from typing import List, Dict, Optional, Tuple, Union
import itertools
import os
import os.path as osp
import re
import json
import pickle
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
X = 'timesteps_total'
SUCC_RATE = 'SuccessRate'
CRASH_RATE = 'CrashRate'
OUT_RATE = 'OutRate'
MAXSTEP_RATE = 'MaxStepRate'
REWARD_MEAN = 'episode_reward_mean' #TODO
all_col = [X, SUCC_RATE, CRASH_RATE, OUT_RATE, MAXSTEP_RATE]
succ_col = [X, SUCC_RATE]
crash_col = [X, CRASH_RATE]
out_col = [X, OUT_RATE]
maxstep_col = [X, MAXSTEP_RATE]
reward_mean_col = [X, REWARD_MEAN]

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
    msg['*']= '*'
    msg['label'] = 're pattern:'
    msg['-']= '-'
    for i, k in enumerate(res):
        msg[f'(label {i+1}) ' + k] = res[k]
    
    if verbose:
        printPanel(msg, title=f'generating total {len(param_space)} param spaces')

    return res


def read_csv_in_dir(
    exp_dir, 
    param_pattern: str, 
    columns: list, 
    seeds: Union[int, List[int]] = None, 
    verbose: bool = False,
) -> List:
    ''' 给定一个实验目录和一个正则表达式匹配式, 筛序出该目录下所有符合该表达式的目录
        并返回其中的progress.csv文件内所需要的列

        如果seed为None: 还会聚集同一个param_pattern的不同种子
        允许输入的pattern也是匹配种子的pattern, 结果不变
        如果指定了seed, 则只读取包含这一个seed或这些seed的目录
    '''
    trial_dirs = [entry for entry in os.listdir(exp_dir) if osp.isdir(osp.join(exp_dir, entry))]
    csv_list = []
    all_seeds = []

    seed_pattern = r'start_seed=(\d*)'

    if isinstance(seeds, int):
        seeds = [seeds]

    if verbose:
        print(f'Reading csv in {exp_dir} for pattern {param_pattern}...')
        trial_dirs = tqdm(trial_dirs)

    for d in trial_dirs:
        # 首先判断目录名是否包含 param_pattern
        if re.search(param_pattern, d):
            # 筛选出目录名中包含种子的的部分: start_seed=*
            match = re.search(seed_pattern, d)
            if match: 
                seed = match.group()[len('start_seed='):]
                if seeds and (int(seed) not in seeds):
                    continue
                all_seeds.append(seed)
                file_path = osp.join(exp_dir, d, 'progress.csv')
                progress = pd.read_csv(file_path)
                data = progress[columns]
                csv_list.append(data)

            # if verbose:
            #     print('found param pattern:', param_pattern)
    
    if verbose:
        print(f'Found {len(all_seeds)} seed (', ', '.join(all_seeds), ')')
    return csv_list


def append_seed_in_dir_name(exp_dir):
    ''' 自动搜索 exp_dir 中的所有 trials 目录中的params.json 文件
        获取其中的start_seed属性, 并将其附加在 trial 目录名后
        用于目录名太长没有start_seed信息
    '''
    trial_dirs = [entry for entry in os.listdir(exp_dir) if osp.isdir(osp.join(exp_dir, entry))]
    
    for t_dir in tqdm(trial_dirs):
        t_dir = osp.join(exp_dir, t_dir)
        # print(t_dir)
        with open(osp.join(t_dir, "params.json"), "r") as f:
            params = json.loads(f.read())
            seed = params['env_config']['start_seed']
        
        os.rename(t_dir, t_dir+'_start_seed='+str(seed))
        # break


def plot_mean_std(
    data, 
    x: str, 
    col: List[str], 
    title=None, 
    lable=None,
    xlabel: bool = True,
    ylabel: bool = True,
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
        param_pattern: str = None, 
        seeds: Union[int, List[int]] = None,
        col=succ_col, 
        title=None, 
        exp_label=None, 
        xlabel: bool = True,
        ylabel: bool = True,
        verbose=False
    ):
    ''' 简单从一个exp目录根据pattern筛选所有trial
        分别读取csv文件, 再拼接, 
        然后按x轴聚集不同种子绘制均值、标准差图
    '''
    if param_pattern == None:
        param_pattern = r'start_seed=(\d*)'

    df_list = read_csv_in_dir(
        exp_dir=exp_dir, 
        param_pattern=param_pattern, 
        columns=col, 
        seeds=seeds, 
        verbose=verbose
    )
    if df_list:
        data = pd.concat(df_list)
    else:
        return

    plot_mean_std(data, X, col, title=title, lable=exp_label, xlabel=xlabel, ylabel=ylabel)


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


def compare_all_metrics_in_one_experiment(
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
    


def compare_all_metrics_for_multi_experiments(
    exps: Dict[str, Union[str, Tuple[str, str]]],
    title: str = '',
):
    ''' 绘制多个实验目录中不同超参设置下的所有指标, 绘制一个2X2的图形
        用于不同目录下不同超参设置下在各个指标下的结果对比
    Args:
        exps: mapping ( exp_name -> exp_info { } )
            exp_info: dict
              ╰───> keys: 
                        'algo_name', 
                        'param_pattern_dict', 
                        'seeds',
                        'label', 如果 param_pattern_dict 未提供，则使用
    '''
    fig = plt.figure(figsize=(12, 8))
    fig.subplots_adjust(top=0.92, hspace=0.25, wspace=0.2)
    
    i = 1
    for metric, col in all_metrics.items(): # 4 metrics
        plt.subplot(2, 2, i)
        xlabel: bool = i >= 3

        for exp_dir, info in exps.items():
            # 获取 param_pattern, 如果没提供，则用None代替
            param_pattern_dict = info.get('param_pattern_dict', {'': None})
            for plabel, pattern in param_pattern_dict.items():
                label = info.get('label', '') 
                if label:
                    label = label + ' ' + plabel
                else:
                    label = plabel

                algo_name = info.get('algo_name', '')
                if algo_name:
                    exp_label = algo_name + ' ' + label

                plot_one_exp(
                    exp_dir=exp_dir, 
                    param_pattern=pattern, 
                    seeds=info.get('seeds', None),
                    col=col, 
                    title=metric, 
                    exp_label=exp_label,
                    xlabel=xlabel, 
                    ylabel=False,
                )
        i += 1

    plt.suptitle(title)


def test_compare_all_metrics_for_multi_experiments():
    exp_dir = 'exp_results/SAPPO_Inter_30agents_v4(share_vf_5e6)[AIBOY]'

    param_pattern_dict = get_param_patthern({
        # 'start_seed': ['5000', '6000', '7000'],
        'vf_share_layers': [['sh', 'sep'], ['True', 'False']],
        # 'vf_share_layers': [['sh'], ['True']],
    }, verbose=True) # {lable -> re pattern}


    exps = {
        exp_dir: dict(
            algo_name='IPPO',
            param_pattern_dict=param_pattern_dict,
            seeds=6000,
        ),
    }

    compare_all_metrics_for_multi_experiments(exps)
    plt.show()



def _sa_exps():
    baseline_dir = 'exp_results/IPPO_Intersection_8seeds_30agents_repeat2'
    ccppo_dir = 'exp_results/CCPPO_Intersection_8seeds_30agents'
    v0_dir = 'exp_results/SAPPO_Inter_30agents_v0'
    v1_dir = 'exp_results/SAPPO_Inter_30agents_v1(reduce_head_input)'
    v2_dir = 'exp_results/SAPPO_Inter_30agents_v2(better_attention)'
    v3_dir = 'exp_results/SAPPO_Inter_30agents_v3(better_attention_layer_norm)'
    v4_dir = 'exp_results/SAPPO_Inter_30agents_v4(share_vf)'
    v4_dir_5e6 = 'exp_results/SAPPO_Inter_30agents_v4(share_vf_5e6)'


    param_pattern_dict = get_param_patthern({
        # 'start_seed': ['5000', '6000', '7000'],
        'vf_share_layers': [['sh', 'sep'], ['True', 'False']],
        # 'vf_share_layers': [['sh'], ['True']],
    }, verbose=False) # {lable -> re pattern}


    param_pattern_dict_ccppo = get_param_patthern({
        'fuse_mode': [['concat', 'mf'], ['concat', 'mf']],
    })

    exps = {
        v0_dir: dict(
            algo_name='SAPPO', 
            label='v0',
        ),
        v3_dir: dict(
            algo_name='SAPPO', 
            label='v3 (upd atn & ln)',
        ),
        v4_dir: dict(
            algo_name='SAPPO', 
            label='v4 (sh vf)',
        ),
        v4_dir_5e6: dict(
            algo_name='SAPPO', 
            label='v4 (sh vf)',
            param_pattern_dict=param_pattern_dict,
            # sees=[8000]
        ),

        ccppo_dir: dict(
            algo_name='CCPPO', 
            param_pattern_dict=param_pattern_dict_ccppo,
            # seeds=[5000, 6000, 7000, 8000],
            # seeds=[5000],
            # label='baseline',
        ),

        baseline_dir: dict(
            algo_name='IPPO', 
            seeds=[5000, 6000, 7000, 8000],
            # seeds=[5000],
            label='baseline',
        ),
    }

    compare_all_metrics_for_multi_experiments(exps)
    plt.show()


if __name__ == "__main__":
    pass

    # baseline_dir = 'exp_results/IPPO_Intersection_8seeds_30agents_repeat2'

    # compare_all_metrics_for_multi_experiments(exps)
    # plt.show()


    # test_compare_all_metrics_for_multi_experiments()

    # _sa_exps()

