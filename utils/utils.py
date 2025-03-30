import os

import torch
import torch.distributed as dist

import math
import numpy as np

from src.common.param import args


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def init_distributed_mode():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        args.DistributedDataParallel = False
        return

    args.DistributedDataParallel = True

    torch.cuda.set_device(gpu)
    print('distributed init (rank {}, word {})'.format(rank, world_size), flush=True)
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank,
    )
    torch.distributed.barrier()


def manual_init_distributed_mode(rank, world_size, local_rank):
    args.DistributedDataParallel = True
    args.batchSize = 1

    gpu = local_rank
    torch.cuda.set_device(gpu)

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(args.DDP_MASTER_PORT)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    os.environ['LOCAL_RANK'] = str(local_rank)

    print('distributed init (rank {}, word {})'.format(rank, world_size), flush=True)
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank,
    )
    torch.distributed.barrier()


def FromPortGetPid(port: int):
    import subprocess
    import time
    import signal

    subprocess_execute = "netstat -nlp | grep {}".format(
        port,
    )

    try:
        p = subprocess.Popen(
            subprocess_execute,
            stdin=None, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            shell=True,
        )
    except Exception as e:
        print(
            "{}\t{}\t{}".format(
                str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
                'FromPortGetPid',
                e,
            )
        )
        return None
    except:
        return None

    pid = None
    for line in iter(p.stdout.readline, b''):
        line = str(line, encoding="utf-8")
        if 'tcp' in line:
            pid = line.strip().split()[-1].split('/')[0]
            try:
                pid = int(pid)
            except:
                pid = None
            break

    try:
        # os.system(("kill -9 {}".format(p.pid)))
        os.kill(p.pid, signal.SIGKILL)
    except:
        pass

    return pid


def non_maximum_suppression_1d(signal, window_size):
    if window_size % 2 == 0:
        raise ValueError("Window size must be odd.")

    half_window = window_size // 2
    length = len(signal)
    result = np.zeros(length)

    for i in range(length):
        start = max(0, i - half_window)
        end = min(length, i + half_window + 1)
        if signal[i] == max(signal[start:end]):
            result[i] = signal[i]

    return result


def calculate_movement_steps(A, B, step_size=5.0):
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)

    distance = np.linalg.norm(B - A)

    steps = math.ceil(distance / step_size)

    direction = B - A
    if distance < 3:
        return 0, [B]
    else:
        unit_direction = direction / distance

    coordinates = [A.copy()]

    for step in range(1, steps + 1):
        current_distance = step * step_size
        if current_distance >= distance:
            current_position = B.copy()
        else:
            current_position = A + unit_direction * current_distance
        coordinates.append(current_position)

    return steps, coordinates