import copy
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import ray
import os
import json
import math
import numpy as np
import random

from attention import AttentionNet
from runner import RLRunner
from parameters import *
from env.task_env import TaskEnv
from scipy.stats import ttest_rel
from torch.distributions import Categorical
import online_reweight


def _extract_state_dict(obj):
    """Accept a full checkpoint dict or a raw state_dict; return the weights state_dict."""
    if isinstance(obj, dict):
        for k in ('best_model', 'model', 'state_dict'):
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
    return obj


# Train on the five task-location distributions in tunable proportions.
# Assignment happens here on the (single) driver at dispatch time, so the exact
# balance is independent of the async order in which Ray workers finish.
LOCATION_DISTS = ['uniform', 'gmm', 'ring', 'bipolar', 'edges']
DIST_WEIGHTS_PATH = os.path.join(SaverParams.MODEL_PATH, 'dist_weights.json')
DIST_STATS_PATH = os.path.join(SaverParams.MODEL_PATH, 'dist_stats.json')


class DistScheduler:
    """Deterministic, exactly-proportional distribution sampler.

    `weights` (a dist->number dict) is turned into integer per-cycle quotas via
    the largest-remainder method; a shuffled "bag" of that exact composition is
    dealt out and refilled when empty. So every full cycle realizes the target
    proportions exactly (not just in expectation). A weight of 0 disables a
    distribution. Equal weights reproduce the one-each-per-5 behaviour."""

    def __init__(self, dists, weights=None):
        self.dists = list(dists)
        self._bag = []
        self.set_weights(weights or {d: 1 for d in self.dists})

    def set_weights(self, weights):
        w = {d: max(0.0, float(weights.get(d, 0))) for d in self.dists}
        if sum(w.values()) <= 0:
            w = {d: 1.0 for d in self.dists}
        self.weights = w
        self.quota = self._largest_remainder(w)
        self._bag = []  # force a refill on the next draw

    @staticmethod
    def _largest_remainder(w):
        positive = {d: v for d, v in w.items() if v > 0}
        total = sum(positive.values())
        if all(abs(v - round(v)) < 1e-9 for v in positive.values()):
            length = int(round(total))                              # integer weights -> exact, minimal cycle
        else:
            length = min(1000, max(len(positive), int(round(total / min(positive.values())))))  # float -> fine resolution
        ideal = {d: length * v / total for d, v in positive.items()}
        quota = {d: int(math.floor(x)) for d, x in ideal.items()}
        remaining = length - sum(quota.values())
        for d in sorted(positive, key=lambda d: ideal[d] - quota[d], reverse=True)[:remaining]:
            quota[d] += 1
        return {d: quota.get(d, 0) for d in w}

    def next(self):
        if not self._bag:
            bag = []
            for d, c in self.quota.items():
                bag += [d] * c
            random.shuffle(bag)
            self._bag = bag
        return self._bag.pop()


def load_dist_weights(path, dists):
    if os.path.exists(path):
        with open(path) as f:
            raw = json.load(f)
        return {d: max(0.0, float(raw.get(d, 0))) for d in dists}
    return None


def save_dist_weights(path, weights):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump({d: round(float(v), 4) for d, v in weights.items()}, f, indent=2)


dist_scheduler = DistScheduler(LOCATION_DISTS)


class Logger(object):
    def __init__(self):
        self.global_net = None
        self.baseline_net = None
        self.optimizer = None
        self.lr_decay = None
        self.writer = SummaryWriter(SaverParams.TRAIN_PATH)
        if SaverParams.SAVE:
            os.makedirs(SaverParams.MODEL_PATH, exist_ok=True)
        if SaverParams.SAVE:
            os.makedirs(SaverParams.GIFS_PATH, exist_ok=True)

    def set(self,  global_net, baseline_net, optimizer, lr_decay):
        self.global_net = global_net
        self.baseline_net = baseline_net
        self.optimizer = optimizer
        self.lr_decay = lr_decay

    def write_to_board(self, tensorboard_data, curr_episode):
        tensorboard_data = np.array(tensorboard_data)
        tensorboard_data = list(np.nanmean(tensorboard_data, axis=0))
        reward, p_l, entropy, grad_norm, success_rate, time, time_cost, waiting, distance, effi = tensorboard_data
        metrics = {'Loss/Learning Rate': self.lr_decay.get_last_lr()[0],
                   'Loss/Policy Loss': p_l,
                   'Loss/Entropy': entropy,
                   'Loss/Grad Norm': grad_norm,
                   'Loss/Reward': reward,
                   'Perf/Makespan': time,
                   'Perf/Success rate': success_rate,
                   'Perf/Time cost': time_cost,
                   'Perf/Waiting time': waiting,
                   'Perf/Traveling distance':distance,
                   'Perf/Waiting Efficiency': effi
                   }
        for k, v in metrics.items():
            self.writer.add_scalar(tag=k, scalar_value=v, global_step=curr_episode)

    def write_dist_stats(self, dist_perf, weights, curr_episode):
        """Per-distribution metrics -> TensorBoard curves + a dist_stats.json the
        LLM (or you) can read to decide new sampling weights."""
        summary = {'episode': curr_episode, 'weights': {d: round(float(w), 4) for d, w in weights.items()}, 'per_dist': {}}
        for d, m in dist_perf.items():
            n = len(m.get('reward', []))
            if n == 0:
                continue
            stats = {'episodes': n}
            for k, vals in m.items():
                if len(vals) == 0:
                    continue
                mean = float(np.nanmean(vals))
                stats[k] = round(mean, 4)
                self.writer.add_scalar(tag=f'Dist/{d}/{k}', scalar_value=mean, global_step=curr_episode)
            summary['per_dist'][d] = stats
        with open(DIST_STATS_PATH, 'w') as f:
            json.dump(summary, f, indent=2)

    def load_saved_model(self):
        ckpt_path = getattr(SaverParams, 'LOAD_CHECKPOINT_PATH', None) or (SaverParams.MODEL_PATH + '/checkpoint.pth')
        print('Loading Model from', ckpt_path)
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        model = 'best_model' if SaverParams.LOAD_FROM == 'best' else 'model'
        # global net = current@20k (continue the trajectory; optimizer stays consistent)
        self.global_net.load_state_dict(checkpoint[model])
        # baseline ("model to beat") = historical best; prefer a separate best file if given,
        # so best-tracking continues against the real best instead of resetting to current.
        best_path = getattr(SaverParams, 'LOAD_BEST_PATH', None)
        if best_path and os.path.exists(best_path):
            self.baseline_net.load_state_dict(_extract_state_dict(torch.load(best_path, map_location='cpu')))
            print('baseline (best) loaded from', best_path)
        else:
            self.baseline_net.load_state_dict(checkpoint.get('best_model', checkpoint[model]))
        if 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'lr_decay' in checkpoint:
            self.lr_decay.load_state_dict(checkpoint['lr_decay'])
        curr_episode = checkpoint.get('episode', 0)
        curr_level = checkpoint.get('level', 0)
        best_perf = checkpoint.get('best_perf', -200)
        dist_weights = checkpoint.get('dist_weights', None)
        print("curr_episode set to ", curr_episode)
        print("best_perf so far is ", best_perf)
        print(self.optimizer.state_dict()['param_groups'][0]['lr'])
        if TrainParams.RESET_OPT:
            self.optimizer = optim.Adam(self.global_net.parameters(), lr=TrainParams.LR)
            self.lr_decay = optim.lr_scheduler.StepLR(self.optimizer, step_size=TrainParams.DECAY_STEP, gamma=0.98)
        return curr_episode, curr_level, best_perf, dist_weights

    def save_model(self, curr_episode, curr_level, best_perf, dist_weights=None):
        print('Saving model', end='\n')
        checkpoint = {"model": self.global_net.state_dict(),
                      "best_model": self.baseline_net.state_dict(),
                      "best_optimizer": self.optimizer.state_dict(),
                      "optimizer": self.optimizer.state_dict(),
                      "episode": curr_episode,
                      "lr_decay": self.lr_decay.state_dict(),
                      "level": curr_level,
                      "best_perf": best_perf,
                      "dist_weights": dist_weights
                      }
        path_checkpoint = "./" + SaverParams.MODEL_PATH + "/checkpoint.pth"
        torch.save(checkpoint, path_checkpoint)
        print('Saved model', end='\n')

    @staticmethod
    def generate_env_params(curr_level=None):
        per_species_num = np.random.randint(EnvParams.SPECIES_AGENTS_RANGE[0], EnvParams.SPECIES_AGENTS_RANGE[1] + 1)
        species_num = np.random.randint(EnvParams.SPECIES_RANGE[0], EnvParams.SPECIES_RANGE[1] + 1)
        tasks_num = np.random.randint(EnvParams.TASKS_RANGE[0], EnvParams.TASKS_RANGE[1] + 1)
        location_dist = dist_scheduler.next()  # one draw per dispatch -> exact target proportions
        params = [(per_species_num, per_species_num), (species_num, species_num), (tasks_num, tasks_num), location_dist]
        return params

    @staticmethod
    def generate_test_set_seed():
        # Each eval instance is a (seed, distribution) pair, balanced across the five
        # distributions, so best-model selection reflects all-distribution performance
        # rather than uniform only. Paired ttest stays valid: the same fixed set is used
        # for both the current and the baseline model until a baseline update.
        seeds = np.random.randint(low=0, high=1e8, size=TrainParams.EVALUATION_SAMPLES).tolist()
        return [(s, LOCATION_DISTS[i % len(LOCATION_DISTS)]) for i, s in enumerate(seeds)]


def fuse_two_dicts(ini_dictionary1, ini_dictionary2):
    if ini_dictionary2 is not None:
        merged_dict = {**ini_dictionary1, **ini_dictionary2}
        final_dict = {}
        for k, v in merged_dict.items():
            final_dict[k] = ini_dictionary1[k] + v
        return final_dict
    else:
        return ini_dictionary1


def main():
    logger = Logger()
    ray.init()
    device = torch.device('cuda') if TrainParams.USE_GPU_GLOBAL else torch.device('cpu')
    local_device = torch.device('cuda') if TrainParams.USE_GPU else torch.device('cpu')

    global_network = AttentionNet(TrainParams.AGENT_INPUT_DIM, TrainParams.TASK_INPUT_DIM, TrainParams.EMBEDDING_DIM).to(device)
    baseline_network = AttentionNet(TrainParams.AGENT_INPUT_DIM, TrainParams.TASK_INPUT_DIM, TrainParams.EMBEDDING_DIM).to(device)
    global_optimizer = optim.Adam(global_network.parameters(), lr=TrainParams.LR)
    lr_decay = optim.lr_scheduler.StepLR(global_optimizer, step_size=TrainParams.DECAY_STEP, gamma=0.98)

    logger.set(global_network, baseline_network, global_optimizer, lr_decay)

    curr_episode = 0
    curr_level = 0
    best_perf = -200
    ckpt_weights = None
    if SaverParams.LOAD_MODEL:
        curr_episode, curr_level, best_perf, ckpt_weights = logger.load_saved_model()

    # distribution sampling weights: file > checkpoint > equal. The json file is the
    # live source of truth (editable mid-run); also mirrored into the checkpoint.
    dist_weights = load_dist_weights(DIST_WEIGHTS_PATH, LOCATION_DISTS)
    if dist_weights is None:
        dist_weights = ckpt_weights if ckpt_weights is not None else {d: 1 for d in LOCATION_DISTS}
        save_dist_weights(DIST_WEIGHTS_PATH, dist_weights)
    dist_scheduler.set_weights(dist_weights)
    print('distribution weights:', dist_scheduler.weights, '-> per-cycle quota:', dist_scheduler.quota)

    # launch meta agents
    meta_agents = [RLRunner.remote(i) for i in range(TrainParams.NUM_META_AGENT)]

    # get initial weights
    if device != local_device:
        weights = global_network.to(local_device).state_dict()
        baseline_weights = baseline_network.to(local_device).state_dict()
        global_network.to(device)
        baseline_network.to(device)
    else:
        weights = global_network.state_dict()
        baseline_weights = baseline_network.state_dict()
    weights_memory = ray.put(weights)
    baseline_weights_memory = ray.put(baseline_weights)

    # launch the first job on each runner
    jobs = []

    for i, meta_agent in enumerate(meta_agents):
        env_params = logger.generate_env_params(curr_level)  # per-worker draw keeps the 5-way balance exact
        jobs.append(meta_agent.training.remote(weights_memory, baseline_weights_memory, curr_episode, env_params))
        curr_episode += 1
    test_set = logger.generate_test_set_seed()
    baseline_value = None
    experience_buffer = {idx:[] for idx in range(7)}
    perf_metrics = {'success_rate': [], 'makespan': [], 'time_cost': [], 'waiting_time': [], 'travel_dist': [], 'efficiency': []}
    dist_perf_keys = list(perf_metrics.keys()) + ['reward']
    dist_perf = {d: {k: [] for k in dist_perf_keys} for d in LOCATION_DISTS}  # rolling per-distribution stats
    training_data = []

    # next episode at which to run the online LLM reweighting loop
    reweight_every = getattr(TrainParams, 'REWEIGHT_EVERY', 0) if getattr(TrainParams, 'ONLINE_REWEIGHT', False) else 0
    next_reweight = (curr_episode // reweight_every + 1) * reweight_every if reweight_every else None

    try:
        while curr_episode < TrainParams.MAX_EPISODE:
            # wait for any job to be completed
            done_id, jobs = ray.wait(jobs)
            done_job = ray.get(done_id)[0]
            buffer, metrics, info = done_job
            experience_buffer = fuse_two_dicts(experience_buffer, buffer)
            perf_metrics = fuse_two_dicts(perf_metrics, metrics)

            # bucket this episode's metrics under the distribution it was sampled from
            episode_dist = info.get('location_dist')
            if episode_dist in dist_perf:
                for k, v in metrics.items():
                    dist_perf[episode_dist][k] += v
                if 'reward' in info:
                    dist_perf[episode_dist]['reward'].append(info['reward'])

            update_done = False
            if len(experience_buffer[0]) >= TrainParams.BATCH_SIZE:
                train_metrics = []
                # env_params = logger.generate_env_params(curr_level)
                while len(experience_buffer[0]) >= TrainParams.BATCH_SIZE:
                    rollouts = {}
                    for k, v in experience_buffer.items():
                        rollouts[k] = v[:TrainParams.BATCH_SIZE]
                    for k in experience_buffer.keys():
                        experience_buffer[k] = experience_buffer[k][TrainParams.BATCH_SIZE:]
                    if len(experience_buffer[0]) < TrainParams.BATCH_SIZE:
                        update_done = True
                    if update_done:
                        for v in experience_buffer.values():
                            del v[:]

                    agent_inputs = torch.stack(rollouts[0], dim=0).to(device)  # (batch,sample_size,2)
                    task_inputs = torch.stack(rollouts[1], dim=0).to(device)  # (batch,sample_size,k_size)
                    action_batch = torch.stack(rollouts[2], dim=0).unsqueeze(1).to(device)  # (batch,1,1)
                    global_mask_batch = torch.stack(rollouts[3], dim=0).to(device)  # (batch,1,1)
                    reward_batch = torch.stack(rollouts[4], dim=0).unsqueeze(1).to(device)  # (batch,1,1)
                    index = torch.stack(rollouts[5]).to(device)
                    advantage_batch = torch.stack(rollouts[6], dim=0).to(device)  # (batch,1,1)

                    # REINFORCE
                    probs, _ = global_network(task_inputs, agent_inputs, global_mask_batch, index)
                    dist = Categorical(probs)
                    logp = dist.log_prob(action_batch.flatten())
                    entropy = dist.entropy().mean()
                    policy_loss = - logp * advantage_batch.flatten().detach()
                    policy_loss = policy_loss.mean()

                    loss = policy_loss
                    global_optimizer.zero_grad()

                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(global_network.parameters(), max_norm=100, norm_type=2)
                    global_optimizer.step()

                    train_metrics.append([reward_batch.mean().item(), policy_loss.item(), entropy.item(), grad_norm.item()])
                lr_decay.step()

                perf_data = []
                for k, v in perf_metrics.items():
                    perf_data.append(np.nanmean(perf_metrics[k]))
                    del v[:]
                train_metrics = np.nanmean(train_metrics, axis=0)
                for v in perf_metrics.values():
                    del v[:]
                data = [*train_metrics, *perf_data]
                training_data.append(data)

            if len(training_data) >= TrainParams.SUMMARY_WINDOW:
                logger.write_to_board(training_data, curr_episode)
                training_data = []

            # get the updated global weights
            if update_done:
                if device != local_device:
                    weights = global_network.to(local_device).state_dict()
                    baseline_weights = baseline_network.to(local_device).state_dict()
                    global_network.to(device)
                    baseline_network.to(device)
                else:
                    weights = global_network.state_dict()
                    baseline_weights = baseline_network.state_dict()
                weights_memory = ray.put(weights)
                baseline_weights_memory = ray.put(baseline_weights)

            env_params = logger.generate_env_params(curr_level)
            jobs.append(meta_agents[info['id']].training.remote(weights_memory, baseline_weights_memory, curr_episode, env_params))
            curr_episode += 1

            if curr_episode // (TrainParams.INCREASE_DIFFICULTY * (curr_level + 1)) == 1 and curr_level < 10:
                curr_level += 1
                print('increase difficulty to level', curr_level)

            if curr_episode % 512 == 0:
                # publish per-distribution stats (TensorBoard + dist_stats.json), then
                # hot-reload sampling weights from the json file if it was edited.
                logger.write_dist_stats(dist_perf, dist_scheduler.weights, curr_episode)
                dist_perf = {d: {k: [] for k in dist_perf_keys} for d in LOCATION_DISTS}  # reset rolling window
                new_weights = load_dist_weights(DIST_WEIGHTS_PATH, LOCATION_DISTS)
                if new_weights is not None and new_weights != dist_scheduler.weights:
                    dist_scheduler.set_weights(new_weights)
                    print('reloaded distribution weights:', dist_scheduler.weights, '-> quota:', dist_scheduler.quota)
                logger.save_model(curr_episode, curr_level, best_perf, dist_scheduler.weights)

            # ---- online LLM reweighting: eval best model on M2-M5 + greedy -> md -> DeepSeek -> new mix ----
            if next_reweight is not None and curr_episode >= next_reweight:
                print(f'[reweight] episode {curr_episode}: evaluating best model on {TrainParams.REWEIGHT_TEST_ROOT}')
                try:
                    new_w, md_path = online_reweight.run(
                        baseline_network, device, TrainParams.REWEIGHT_TEST_ROOT, curr_episode,
                        dist_scheduler.weights, SaverParams.MODEL_PATH, TrainParams.DEEPSEEK_MODEL,
                        floor=TrainParams.REWEIGHT_FLOOR, max_change=TrainParams.REWEIGHT_MAX_CHANGE)
                    print(f'[reweight] report written to {md_path}')
                    if new_w:
                        dist_scheduler.set_weights(new_w)
                        save_dist_weights(DIST_WEIGHTS_PATH, new_w)
                        print('[reweight] new mix:', dist_scheduler.weights, '-> quota:', dist_scheduler.quota)
                except Exception as exc:
                    print('[reweight] failed, keeping current mix:', exc)
                next_reweight += reweight_every

            if TrainParams.EVALUATE:
                if curr_episode % 1024 == 0:
                    # stop the training
                    ray.wait(jobs, num_returns=TrainParams.NUM_META_AGENT)
                    for a in meta_agents:
                        ray.kill(a)
                    print('Evaluate baseline model at ', curr_episode)

                    # test the baseline model on the new test set
                    if baseline_value is None:
                        test_agent_list = [RLRunner.remote(metaAgentID=i) for i in range(TrainParams.NUM_META_AGENT)]
                        for _, test_agent in enumerate(test_agent_list):
                            ray.get(test_agent.set_baseline_weights.remote(baseline_weights_memory))
                        rewards = dict()
                        seed_list = copy.deepcopy(test_set)
                        evaluate_jobs = []
                        for i in range(TrainParams.NUM_META_AGENT):
                            s, d = seed_list.pop()
                            evaluate_jobs.append(test_agent_list[i].testing.remote(seed=s, location_dist=d))
                        while True:
                            test_done_id, evaluate_jobs = ray.wait(evaluate_jobs)
                            test_result = ray.get(test_done_id)[0]
                            reward, seed, meta_id = test_result
                            rewards[seed] = reward
                            if seed_list:
                                s, d = seed_list.pop()
                                evaluate_jobs.append(test_agent_list[meta_id].testing.remote(seed=s, location_dist=d))
                            if len(rewards) == TrainParams.EVALUATION_SAMPLES:
                                break
                        rewards = dict(sorted(rewards.items()))
                        baseline_value = np.stack(list(rewards.values()))
                        for a in test_agent_list:
                            ray.kill(a)

                    # test the current model's performance
                    test_agent_list = [RLRunner.remote(metaAgentID=i) for i in range(TrainParams.NUM_META_AGENT)]
                    for _, test_agent in enumerate(test_agent_list):
                        ray.get(test_agent.set_baseline_weights.remote(weights_memory))
                    rewards = dict()
                    seed_list = copy.deepcopy(test_set)
                    evaluate_jobs = []
                    for i in range(TrainParams.NUM_META_AGENT):
                        s, d = seed_list.pop()
                        evaluate_jobs.append(test_agent_list[i].testing.remote(seed=s, location_dist=d))
                    while True:
                        test_done_id, evaluate_jobs = ray.wait(evaluate_jobs)
                        test_result = ray.get(test_done_id)[0]
                        reward, seed, meta_id = test_result
                        rewards[seed] = reward
                        if seed_list:
                            s, d = seed_list.pop()
                            evaluate_jobs.append(test_agent_list[meta_id].testing.remote(seed=s, location_dist=d))
                        if len(rewards) == TrainParams.EVALUATION_SAMPLES:
                            break
                    rewards = dict(sorted(rewards.items()))
                    test_value = np.stack(list(rewards.values()))
                    for a in test_agent_list:
                        ray.kill(a)

                    meta_agents = [RLRunner.remote(i) for i in range(TrainParams.NUM_META_AGENT)]

                    # update baseline if the model improved more than 5%
                    print('test value', test_value.mean())
                    print('baseline value', baseline_value.mean())
                    if test_value.mean() > baseline_value.mean():
                        _, p = ttest_rel(test_value, baseline_value)
                        print('p value', p)
                        if p < 0.05:
                            print('update baseline model at ', curr_episode)
                            if device != local_device:
                                weights = global_network.to(local_device).state_dict()
                                global_network.to(device)
                            else:
                                weights = global_network.state_dict()
                            baseline_weights = copy.deepcopy(weights)
                            baseline_network.load_state_dict(baseline_weights)
                            weights_memory = ray.put(weights)
                            baseline_weights_memory = ray.put(baseline_weights)
                            test_set = logger.generate_test_set_seed()
                            print('update test set')
                            baseline_value = None
                            best_perf = test_value.mean()
                            logger.save_model(curr_episode, None, best_perf, dist_scheduler.weights)
                    jobs = []
                    for i, meta_agent in enumerate(meta_agents):
                        env_params = logger.generate_env_params(curr_level)  # per-worker draw keeps the 5-way balance exact
                        jobs.append(meta_agent.training.remote(weights_memory, baseline_weights_memory, curr_episode, env_params))
                        curr_episode += 1

        # reached the episode cap -> persist a final checkpoint and shut down cleanly
        print(f'reached MAX_EPISODE = {TrainParams.MAX_EPISODE} at episode {curr_episode}; saving final model')
        logger.save_model(curr_episode, curr_level, best_perf, dist_scheduler.weights)
        for a in meta_agents:
            ray.kill(a)

    except KeyboardInterrupt:
        print("CTRL_C pressed. Killing remote workers")
        logger.save_model(curr_episode, curr_level, best_perf, dist_scheduler.weights)
        for a in meta_agents:
            ray.kill(a)


if __name__ == "__main__":
    main()
