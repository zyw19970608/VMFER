import os
import json
import psutil
import torch

def get_pgid_pid_count(pgid):
    count = 0
    for proc in psutil.process_iter(['pid', 'ppid', 'name']):
        try:
            if os.getpgid(proc.pid) == pgid:
                count += 1
        except (psutil.NoSuchProcess, ProcessLookupError):
            pass
    return count

def all_pgid_num(pgids):
    _source_process_num = 0
    for temp in pgids:
        _source_process_num += get_pgid_pid_count(temp)
    return _source_process_num

def load_config(config_file):
    with open(config_file, 'r') as file:
        return json.load(file)

def main():
    config = load_config('config.json')
    temp = 0
    count = 0
    order_list = []
    gpu_num = torch.cuda.device_count()

    for env in config['environments']:
        dirname = env + '_vmfer'
        os.makedirs('./data/' + dirname, exist_ok=True)

        for algo in config['algorithms']:
            for vmfflag in config['vmf_flags']:
                for seed in config['seeds']:
                    if vmfflag == 'default_autoupbound':
                        order = f'CUDA_VISIBLE_DEVICES={temp % gpu_num} nohup python -u {config["run_file"]} --env_name={env} '
                        order += f'--seed={seed} --algo={algo} '
                        order += f'--experiment_name={algo}_vmf_{vmfflag}_ver '
                        order += f'--VMFActorUpdate={vmfflag} --wandb '
                        order += f'>./data/{dirname}/{algo}_vmf_{vmfflag}_{seed}_ver.log 2>&1 &'
                        count += 1
                        temp += 1
                        print(f'{count}: {order}')
                        order_list.append(order)
                    else:
                        for per in config['PER']:
                            for rt in config['refill_time']:
                                order = f'CUDA_VISIBLE_DEVICES={(temp + 0) % gpu_num} nohup python -u {config["run_file"]} --env_name={env} '
                                order += f'--seed={seed} --algo={algo} --refill_time={rt} --PER={per} '
                                order += f'--experiment_name={algo}_vmf_{vmfflag}_rt{rt}_PER{per}_wrj_ver '
                                order += f'--VMFActorUpdate={vmfflag} --wandb '
                                order += f'>./data/{dirname}/{algo}_vmf_{vmfflag}_{seed}_rt{rt}_PER{per}_wrj_ver.log 2>&1 &'
                                count += 1
                                temp += 1
                                print(f'{count}: {order}')
                                order_list.append(order)

    print(order_list)
    pgids = config['pgids']
    source_process_num = all_pgid_num(pgids)

    while order_list:
        temp_process_num = all_pgid_num(pgids)

        if temp_process_num < source_process_num:
            for _ in range(source_process_num - temp_process_num):
                if order_list:
                    print(order_list[-1])
                    os.system(order_list.pop())
            source_process_num = temp_process_num
        if temp_process_num == 0 and order_list:
            print(order_list[-1])
            os.system(order_list.pop())

if __name__ == '__main__':
    main()
