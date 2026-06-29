import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.animation import FuncAnimation
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from itertools import combinations, product
import copy


class TaskEnv:
    def __init__(self, per_species_range=(10, 10), species_range=(5, 5), tasks_range=(30, 30), traits_dim=5,
                 decision_dim=10, max_task_size=2, duration_scale=5, seed=None, plot_figure=False,
                 location_dist='uniform'):
        """
        :param traits_dim: number of capabilities in this problem, e.g. 3 traits
        :param seed: seed to generate pseudo random problem instance
        :param location_dist: spatial distribution of task locations. One of
            'uniform', 'gmm', 'thomas', 'poisson_disk', 'ring'. Only the task
            positions change; every other part of the instance is unaffected.
        """
        self.rng = None
        self.location_dist = location_dist
        self.per_species_range = per_species_range
        self.species_range = species_range
        self.tasks_range = tasks_range
        self.max_task_size = max_task_size
        self.duration_scale = duration_scale
        self.plot_figure = plot_figure
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.traits_dim = traits_dim
        self.decision_dim = decision_dim

        self.task_dic, self.agent_dic, self.depot_dic, self.species_dict = self.generate_env()
        self.species_distance_matrix, self.species_neighbor_matrix = self.generate_distance_matrix()
        # self.species_mask = self.calculate_optimized_ability()
        self.tasks_num = len(self.task_dic)
        self.agents_num = len(self.agent_dic)
        self.species_num = len(self.species_dict['number'])
        self.coalition_matrix = np.zeros((self.agents_num, self.tasks_num))
        # self.best_route = self.calculate_tsp_route()

        self.current_time = 0
        self.dt = 0.1
        self.max_waiting_time = 200
        self.depot_waiting_time = 0
        self.finished = False
        self.reactive_planning = False

    def random_int(self, low, high, size=None):
        if self.rng is not None:
            integer = self.rng.integers(low, high, size)
        else:
            integer = np.random.randint(low, high, size)
        return integer

    def random_value(self, row, col):
        if self.rng is not None:
            value = self.rng.random((row, col))
        else:
            value = np.random.rand(row, col)
        return value

    def random_choice(self, a, size=None, replace=True):
        if self.rng is not None:
            choice = self.rng.choice(a, size, replace)
        else:
            choice = np.random.choice(a, size, replace)
        return choice

    def random_normal(self, loc=0.0, scale=1.0, size=None):
        if self.rng is not None:
            return self.rng.normal(loc, scale, size)
        return np.random.normal(loc, scale, size)

    def generate_locations(self, num, dist=None):
        """Sample `num` 2D points in the unit square [0, 1]^2 according to `dist`.
        Only the spatial layout changes; every other part of the instance is unaffected.

        The five options are ordered by how spread out the points are, which is what
        the makespan objective actually responds to (greedy-baseline makespan at n=30,
        velocity=0.2, duration_scale=5, for reference):
            'gmm'     compact clusters        ~26   (shortest)
            'ring'    points on a ring        ~32
            'uniform' baseline                ~35
            'bipolar' two distant groups      ~38
            'edges'   points on the borders   ~41   (longest)
        """
        dist = self.location_dist if dist is None else dist
        if dist == 'uniform':
            locs = self.random_value(num, 2)
        elif dist == 'gmm':
            locs = self._gmm_locations(num)
        elif dist == 'ring':
            locs = self._ring_locations(num)
        elif dist == 'bipolar':
            locs = self._bipolar_locations(num)
        elif dist == 'edges':
            locs = self._edge_locations(num)
        else:
            raise ValueError(f'Unknown location_dist: {dist}')
        return np.clip(locs, 0.0, 1.0)

    def _gmm_locations(self, num, n_clusters=None, sigma=0.045):
        # Tight attracting hotspots: 2-3 cluster centers, points scatter normally around them.
        if n_clusters is None:
            n_clusters = int(self.random_int(2, 4))  # 2-3 clusters
        centers = 0.2 + 0.6 * self.random_value(n_clusters, 2)  # keep centers off the border
        assign = self.random_int(0, n_clusters, num)
        return centers[assign] + self.random_normal(0.0, sigma, (num, 2))

    def _ring_locations(self, num, noise=0.02):
        # Structured: points lie on 1-2 concentric circles around the map center.
        n_rings = int(self.random_int(1, 3))  # 1-2 rings
        radii = np.linspace(0.3, 0.4, n_rings)
        center = np.array([0.5, 0.5])
        ring_idx = self.random_int(0, n_rings, num)
        angles = self.random_value(num, 1).reshape(-1) * 2 * np.pi
        rad = radii[ring_idx] + self.random_normal(0.0, noise, num)
        x = center[0] + rad * np.cos(angles)
        y = center[1] + rad * np.sin(angles)
        return np.stack([x, y], axis=1)

    def _bipolar_locations(self, num, sigma=0.05):
        # Two distant groups near opposite corners -> agents must shuttle across the map.
        centers = np.array([[0.15, 0.15], [0.85, 0.85]])
        assign = self.random_int(0, 2, num)
        return centers[assign] + self.random_normal(0.0, sigma, (num, 2))

    def _edge_locations(self, num):
        # Points pushed onto the four borders -> maximum spatial extent / longest tours.
        pts = self.random_value(num, 2)
        side = self.random_int(0, 4, num)
        pts[side == 0, 0] = 0.02
        pts[side == 1, 0] = 0.98
        pts[side == 2, 1] = 0.02
        pts[side == 3, 1] = 0.98
        return pts

    def generate_task(self, tasks_num):
        tasks_ini = self.random_int(0, self.max_task_size, (tasks_num, self.traits_dim))
        while not np.all(np.sum(tasks_ini, axis=1) != 0):
            tasks_ini = self.random_int(0, self.max_task_size, (tasks_num, self.traits_dim))
        # tasks_ini = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [2, 1, 0, 0, 0], [1, 1, 0, 0, 0], [0, 2, 0, 0, 0], [1, 1, 0, 0, 0]])
        return tasks_ini

    def generate_agent(self, species_num):
        # agents_ini = self.random_value(species_num, self.traits_dim) > 0.8
        # while not np.all(np.sum(agents_ini, axis=1) != 0):
        #     agents_ini = self.random_value(species_num, self.traits_dim) > 0.8

        agents_ini = self.random_int(0, 2, (species_num, self.traits_dim))
        while not np.all(np.sum(agents_ini, axis=1) != 0) or np.unique(agents_ini, axis=0).shape[0] != species_num:
            agents_ini = self.random_int(0, 2, (species_num, self.traits_dim))

        # agents_ini = np.diag(np.ones(self.traits_dim))
        # agents_ini = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]])
        return agents_ini

    def generate_env(self):
        tasks_num = self.random_int(self.tasks_range[0], self.tasks_range[1] + 1)
        species_num = self.random_int(self.species_range[0], self.species_range[1] + 1)
        agents_species_num = [self.random_int(self.per_species_range[0], self.per_species_range[1] + 1) for _ in range(species_num)]

        agents_ini = self.generate_agent(species_num)
        tasks_ini = self.generate_task(tasks_num)
        while not np.all(np.matmul(agents_species_num, agents_ini) >= tasks_ini):
            agents_ini = self.generate_agent(species_num)
            tasks_ini = self.generate_task(tasks_num)

        depot_loc = self.random_value(species_num, 2)
        cost_ini = [self.random_value(1, 1) for _ in range(species_num)]
        tasks_loc = self.generate_locations(tasks_num)  # spatial distribution controlled by self.location_dist
        tasks_time = self.random_value(tasks_num, 1) * self.duration_scale

        task_dic = dict()
        agent_dic = dict()
        depot_dic = dict()
        species_dict = dict()
        species_dict['abilities'] = agents_ini
        species_dict['number'] = agents_species_num

        for i in range(tasks_num):
            task_dic[i] = {'ID': i,
                           'requirements': tasks_ini[i, :],  # requirements of the task
                           'members': [],  # members of the task
                           'cost': [],  # cost of each agent
                           'location': tasks_loc[i, :],  # location of the task
                           'feasible_assignment': False,  # whether the task assignment is feasible
                           'finished': False,
                           'time_start': 0,
                           'time_finish': 0,
                           'status': tasks_ini[i, :],
                           'time': float(tasks_time[i, :]),
                           'sum_waiting_time': 0,
                           'efficiency': 0,
                           'abandoned_agent': [],
                           'optimized_ability': None,
                           'optimized_species': []}

        i = 0
        for s, n in enumerate(agents_species_num):
            species_dict[s] = []
            for j in range(n):
                agent_dic[i] = {'ID': i,
                                'species': s,
                                'abilities': agents_ini[s, :],
                                'location': depot_loc[s, :],
                                'route': [- s - 1],
                                'current_task': - s - 1,
                                'contributed': False,
                                'arrival_time': [0.],
                                'cost': cost_ini[s],
                                'travel_time': 0,
                                'velocity': 0.2,
                                'next_decision': 0,
                                'depot': depot_loc[s, :],
                                'travel_dist': 0,
                                'sum_waiting_time': 0,
                                'current_action_index': 0,
                                'decision_step': 0,
                                'task_waiting_ratio': 1,
                                'trajectory': [],
                                'angle': 0,
                                'returned': False,
                                'assigned': False,
                                'pre_set_route': None,
                                'no_choice': False}
                species_dict[s].append(i)
                i += 1

        for s in range(species_num):
            depot_dic[s] = {'location': depot_loc[s, :],
                            'members': species_dict[s],
                            'ID': - s - 1}

        return task_dic, agent_dic, depot_dic, species_dict

    def generate_distance_matrix(self):
        species_distance_matrix = {}
        species_neighbor_matrix = {}
        for species in range(len(self.species_dict['number'])):
            tmp_dic = {-1: self.depot_dic[species], **self.task_dic}
            distances = {}
            for from_counter, from_node in tmp_dic.items():
                distances[from_counter] = {}
                for to_counter, to_node in tmp_dic.items():
                    if from_counter == to_counter:
                        distances[from_counter][to_counter] = 0
                    else:
                        distances[from_counter][to_counter] = self.calculate_eulidean_distance(from_node, to_node)

            sorted_distance_matrix = {k: sorted(dist, key=lambda x: dist[x]) for k, dist in distances.items()}
            species_distance_matrix[species] = distances
            species_neighbor_matrix[species] = sorted_distance_matrix
        return species_distance_matrix, species_neighbor_matrix

    def reset(self, test_env=None, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = None
        if test_env is not None:
            self.task_dic, self.agent_dic, self.depot_dic, self.species_dict = test_env
        else:
            self.task_dic, self.agent_dic, self.depot_dic, self.species_dict = self.generate_env()
        self.tasks_num = len(self.task_dic)
        self.agents_num = len(self.agent_dic)
        self.species_num = len(self.species_dict['number'])
        self.coalition_matrix = np.zeros((self.agents_num, self.tasks_num))
        self.current_time = 0
        self.finished = False

    def init_state(self):
        for task in self.task_dic.values():
            task.update(members=[], cost=[], finished=False, status=task['requirements'],feasible_assignment=False,
                        time_start=0, time_finish=0, sum_waiting_time=0, efficiency=0, abandoned_agent=[])
        for agent in self.agent_dic.values():
            agent.update(route=[-agent['species'] - 1], location=agent['depot'], contributed=False,
                         next_decision=0, travel_time=0, travel_dist=0, arrival_time=[0.], assigned=False,
                         sum_waiting_time=0, current_action_index=0, decision_step=0, trajectory=[], angle=0,
                         returned=False, pre_set_route=None, current_task=-1, task_waiting_ratio=1, no_choice=False, next_action=0)
        for depot in self.depot_dic.values():
            depot.update(members=self.species_dict[-depot['ID'] - 1])
        self.current_time = 0
        self.max_waiting_time = 200
        self.finished = False

    @staticmethod
    def find_by_key(data, target):
        for key, value in data.items():
            if isinstance(value, dict):
                yield from TaskEnv.find_by_key(value, target)
            elif key == target:
                yield value

    @staticmethod
    def get_matrix(dictionary, key):
        """
        :param key: the key to index
        :param dictionary: the dictionary for key to index
        """
        key_matrix = []
        for value in dictionary.values():
            key_matrix.append(value[key])
        return key_matrix

    @staticmethod
    def calculate_eulidean_distance(agent, task):
        return np.linalg.norm(agent['location'] - task['location'])

    def calculate_optimized_ability(self):
        for task in self.task_dic.values():
            task_status = task['status']
            # find all possible combinations of the group
            in_species_num = self.species_dict['number']
            species_ability = self.species_dict['abilities']
            num_set = [list(range(0, self.max_task_size + 1)) for _ in in_species_num]
            group_combinations = list(product(*num_set))

            abilities = []
            contained_spe = []
            for sample in group_combinations:
                ability = np.zeros((1, self.traits_dim))
                for spe, num in enumerate(sample):
                    ability += sample[spe] * species_ability[spe]
                contained_spe.append(np.array(sample) > 0)
                abilities.append(ability)

            effective_ability = np.maximum(np.minimum(task_status, np.vstack(abilities)), 0)
            score = np.divide(effective_ability, np.vstack(abilities), where=np.vstack(abilities) > 0,
                              out=np.zeros_like(np.vstack(abilities), dtype=float)) * effective_ability
            score = np.sum(score, axis=1)
            action_index = np.argmax(score)
            group_sort = np.argsort(score)[-2:]
            task['optimized_ability'] = abilities[action_index]
            optimized_species = []
            for ind in group_sort:
                optimized_species.append(contained_spe[ind])
            task['optimized_species'] = np.logical_or(*optimized_species)
        species_mask = np.vstack(self.get_matrix(self.task_dic, 'optimized_species'))
        return species_mask

    def get_current_agent_status(self, agent):
        status = []
        for a in self.agent_dic.values():
            if a['current_task'] >= 0:
                current_task = a['current_task']
                arrival_time = self.get_arrival_time(a['ID'], current_task)
                travel_time = np.clip(arrival_time - self.current_time, a_min=0, a_max=None)
                if self.current_time <= self.task_dic[current_task]['time_start']:
                    current_waiting_time = np.clip(self.current_time - arrival_time, a_min=0, a_max=None)
                    remaining_working_time = np.clip(self.task_dic[current_task]['time_start'] + self.task_dic[current_task]['time'] - self.current_time, a_min=0, a_max=None)
                else:
                    current_waiting_time = 0
                    remaining_working_time = 0
            else:
                travel_time = 0
                current_waiting_time = 0
                remaining_working_time = 0
            temp_status = np.hstack([a['abilities'], travel_time, remaining_working_time, current_waiting_time,
                                     agent['location'] - a['location'], a['assigned']])
            status.append(temp_status)
        current_agents = np.vstack(status)
        return current_agents

    def get_current_task_status(self, agent):
        status = []
        for t in self.task_dic.values():
            travel_time = self.calculate_eulidean_distance(agent, t) / agent['velocity']
            temp_status = np.hstack([t['status'], t['requirements'], t['time'], travel_time,
                                     agent['location'] - t['location'], t['feasible_assignment']])
            status.append(temp_status)
        status = [np.hstack([np.zeros(self.traits_dim), - np.ones(self.traits_dim), 0,
                             self.calculate_eulidean_distance(agent,
                                                              self.depot_dic[agent['species']])
                             / agent['velocity'], agent['location'] - agent['depot'], 1])] + status
        current_tasks = np.vstack(status)
        return current_tasks

    def get_unfinished_task_mask(self):
        mask = np.logical_not(self.get_unfinished_tasks())
        return mask

    def get_unfinished_tasks(self):
        unfinished_tasks = []
        for task in self.task_dic.values():
            unfinished_tasks.append(task['feasible_assignment'] is False and np.any(task['status'] > 0))
        return unfinished_tasks

    def get_arrival_time(self, agent_id, task_id):
        arrival_time = self.agent_dic[agent_id]['arrival_time']
        arrival_for_task = np.where(np.array(self.agent_dic[agent_id]['route']) == task_id)[0][-1]
        return float(arrival_time[arrival_for_task])

    def get_abilities(self, members):
        if len(members) == 0:
            return np.zeros(self.traits_dim)
        else:
            return np.sum(np.array([self.agent_dic[member]['abilities'] for member in members]), axis=0)

    def get_contributable_task_mask(self, agent_id):
        agent = self.agent_dic[agent_id]
        contributable_task_mask = np.ones(self.tasks_num, dtype=bool)
        for task in self.task_dic.values():
            if not task['feasible_assignment']:
                ability = np.maximum(np.minimum(task['status'], agent['abilities']), 0.)
                if ability.sum() > 0:
                    contributable_task_mask[task['ID']] = False
        return contributable_task_mask

    def get_waiting_tasks(self):
        waiting_tasks = np.ones(self.tasks_num, dtype=bool)
        waiting_agents = []
        for task in self.task_dic.values():
            if not task['feasible_assignment'] and len(task['members']) > 0:
                waiting_tasks[task['ID']] = False
                waiting_agents += task['members']
        return waiting_tasks, waiting_agents

    def agent_update(self):
        for agent in self.agent_dic.values():
            if agent['current_task'] < 0:
                if np.all(self.get_matrix(self.task_dic, 'feasible_assignment')):
                    agent['next_decision'] = np.nan
                elif not np.isnan(agent['next_decision']):
                    agent['next_decision'] = np.inf
                else:
                    pass
            else:
                current_task = self.task_dic[agent['current_task']]
                if current_task['feasible_assignment']:
                    if agent['ID'] in current_task['members']:
                        agent['next_decision'] = float(current_task['time_finish'])
                        if self.current_time >= float(current_task['time_start']):
                            agent['assigned'] = True
                    else:
                        agent['next_decision'] = self.get_arrival_time(agent['ID'], current_task['ID']) + self.max_waiting_time
                        agent['assigned'] = False
                else:
                    agent['next_decision'] = self.get_arrival_time(agent['ID'], current_task['ID']) + self.max_waiting_time
                    agent['assigned'] = False

    def task_update(self):
        f_task = []
        # check each task status and whether it is finished
        for task in self.task_dic.values():
            if not task['feasible_assignment']:
                abilities = self.get_abilities(task['members'])
                arrival = np.array([self.get_arrival_time(member, task['ID']) for member in task['members']])
                task['status'] = task['requirements'] - abilities  # update task status
                # Agents will wait for the other agents to arrive
                if (task['status'] <= 0).all():
                    if np.max(arrival) - np.min(arrival) <= self.max_waiting_time:
                        task['time_start'] = float(np.max(arrival, keepdims=True))
                        task['time_finish'] = float(np.max(arrival, keepdims=True) + task['time'])
                        task['feasible_assignment'] = True
                        f_task.append(task['ID'])
                    else:
                        task['feasible_assignment'] = False
                        infeasible_members = arrival <= np.max(arrival, keepdims=True) - self.max_waiting_time
                        for member in np.array(task['members'])[infeasible_members]:
                            task['members'].remove(member)
                            task['abandoned_agent'].append(member)
                else:
                    task['feasible_assignment'] = False
                    for member in np.array(task['members']):
                        if self.current_time - self.get_arrival_time(member, task['ID']) >= self.max_waiting_time:
                            task['members'].remove(member)
                            task['abandoned_agent'].append(member)
            else:
                if self.current_time >= task['time_finish']:
                    task['finished'] = True

        # check depot status
        for depot in self.depot_dic.values():
            for member in depot['members']:
                if self.current_time >= self.get_arrival_time(member, depot['ID']) and np.all(self.get_matrix(self.task_dic, 'feasible_assignment')):
                    self.agent_dic[member]['returned'] = True
        return f_task

    def next_decision(self):
        decision_time = np.array(self.get_matrix(self.agent_dic, 'next_decision'))
        if np.all(np.isnan(decision_time)):
            return ([], []), max(map(lambda x: max(x) if x else 0, self.get_matrix(self.agent_dic, 'arrival_time')))
        no_choice = self.get_matrix(self.agent_dic, 'no_choice')
        decision_time = np.where(no_choice, np.inf, decision_time)
        next_decision = np.nanmin(decision_time)
        if np.isinf(next_decision):
            arrival_time = np.array([agent['arrival_time'][-1] for agent in self.agent_dic.values()])
            decision_time = np.where(no_choice, np.inf, arrival_time)
            next_decision = np.nanmin(decision_time)
        finished_agents = np.where(decision_time == next_decision)[0].tolist()
        blocked_agents = []
        for agent_id in np.where(np.isinf(decision_time))[0].tolist():
            if next_decision >= self.agent_dic[agent_id]['arrival_time'][-1]:
                blocked_agents.append(agent_id)
        release_agents = (finished_agents, blocked_agents)
        return release_agents, next_decision

    def agent_step(self, agent_id, task_id, decision_step):
        """
        :param agent_id: the id of agent
        :param task_id: the id of task
        :param decision_step: the decision step of the agent
        :return: end_episode, finished_tasks
        """
        #  choose any task
        task_id = task_id - 1
        if task_id != -1:
            agent = self.agent_dic[agent_id]
            task = self.task_dic[task_id]
            if task['feasible_assignment']:
                return -1, False, []
        else:
            agent = self.agent_dic[agent_id]
            task = self.depot_dic[agent['species']]
        agent['route'].append(task['ID'])
        previous_task = agent['current_task']
        agent['current_task'] = task_id
        travel_time = self.calculate_eulidean_distance(agent, task) / agent['velocity']
        agent['travel_time'] = travel_time
        agent['travel_dist'] += self.calculate_eulidean_distance(agent, task)
        if previous_task >= 0 and self.task_dic[previous_task]['time_finish'] < self.current_time:
            current_time = self.task_dic[previous_task]['time_finish']
        else:
            current_time = self.current_time
        agent['arrival_time'] += [current_time + travel_time]
        # calculate the angle from current location to next location
        agent['location'] = task['location']
        agent['decision_step'] = decision_step
        agent['no_choice'] = False

        if agent_id not in task['members']:
            task['members'].append(agent_id)
        f_t = self.task_update()
        self.agent_update()
        return 0, True, f_t

    def agent_observe(self, agent_id, max_waiting=False):
        agent = self.agent_dic[agent_id]
        mask = self.get_unfinished_task_mask()
        contributable_mask = self.get_contributable_task_mask(agent_id)
        mask = np.logical_or(mask, contributable_mask)
        if max_waiting:
            waiting_tasks_mask, waiting_agents = self.get_waiting_tasks()
            waiting_len = np.sum(waiting_tasks_mask == 0)
            if waiting_len > 5:
                mask = np.logical_or(mask, waiting_tasks_mask)
        mask = np.insert(mask, 0, False)
        # if mask.all():
        #     mask = np.insert(mask, 0, False)
        # else:
        #     mask = np.insert(mask, 0, True)
        agents_info = np.expand_dims(self.get_current_agent_status(agent), axis=0)
        tasks_info = np.expand_dims(self.get_current_task_status(agent), axis=0)
        mask = np.expand_dims(mask, axis=0)
        return tasks_info, agents_info, mask

    def calculate_waiting_time(self):
        for agent in self.agent_dic.values():
            agent['sum_waiting_time'] = 0
        for task in self.task_dic.values():
            arrival = np.array([self.get_arrival_time(member, task['ID']) for member in task['members']])
            if len(arrival) != 0:
                if task['feasible_assignment']:
                    task['sum_waiting_time'] = np.sum(np.max(arrival) - arrival) \
                                               + len(task['abandoned_agent']) * self.max_waiting_time
                else:
                    task['sum_waiting_time'] = np.sum(self.current_time - arrival) \
                                               + len(task['abandoned_agent']) * self.max_waiting_time
            else:
                task['sum_waiting_time'] = len(task['abandoned_agent']) * self.max_waiting_time
            for member in task['members']:
                if task['feasible_assignment']:
                    self.agent_dic[member]['sum_waiting_time'] += np.max(arrival) - self.get_arrival_time(member, task['ID'])
                else:
                    self.agent_dic[member]['sum_waiting_time'] += self.current_time - self.get_arrival_time(member, task['ID']) if self.current_time - self.get_arrival_time(member, task['ID']) > 0 else 0
            for member in task['abandoned_agent']:
                self.agent_dic[member]['sum_waiting_time'] += self.max_waiting_time

    def check_finished(self):
        self.task_update()
        decision_agents, current_time = self.next_decision()
        # dead_lock = self.check_dead_lock()
        if len(decision_agents[0]) + len(decision_agents[1]) == 0:
            self.current_time = current_time
            finished = np.all(self.get_matrix(self.agent_dic, 'returned')) and np.all(self.get_matrix(self.task_dic, 'finished'))
        else:
            finished = False
        return finished

    def generate_traj(self):
        for agent in self.agent_dic.values():
            # save the location of the agent as trajectory
            time_step = 0
            angle = 0
            for i in range(1, len(agent['route'])):
                current_task = self.task_dic[agent['route'][i - 1]] if agent['route'][i - 1] >= 0 else self.depot_dic[agent['species']]
                next_task = self.task_dic[agent['route'][i]] if agent['route'][i] >= 0 else self.depot_dic[agent['species']]
                angle = np.arctan2(next_task['location'][1] - current_task['location'][1],
                                   next_task['location'][0] - current_task['location'][0])
                distance = self.calculate_eulidean_distance(next_task, current_task)
                total_time = distance / agent['velocity']
                arrival_time_next = agent['arrival_time'][i]
                arrival_time_current = agent['arrival_time'][i - 1]
                if next_task['ID'] >= 0 and agent['ID'] in next_task['members'] \
                        and next_task['feasible_assignment']:
                    if next_task['time_start'] - arrival_time_next <= self.max_waiting_time:
                        next_decision = next_task['time_finish']
                    else:
                        next_decision = arrival_time_next + self.max_waiting_time
                elif next_task['ID'] < 0 and i != len(agent['route']) - 1:
                    next_decision = arrival_time_next + self.depot_waiting_time
                else:
                    next_decision = arrival_time_next + self.max_waiting_time
                if current_task['ID'] < 0 and i == 1:
                    current_decision = 0
                elif current_task['ID'] < 0:
                    current_decision = arrival_time_current + self.depot_waiting_time
                else:
                    if agent['ID'] in current_task['members'] \
                            and current_task['time_start'] - arrival_time_current <= self.max_waiting_time \
                            and current_task['feasible_assignment']:
                        current_decision = current_task['time_finish']
                    else:
                        current_decision = arrival_time_current + self.max_waiting_time
                while time_step < next_decision:
                    time_step += self.dt
                    if time_step < arrival_time_next:
                        fraction_of_time = (time_step - current_decision) / total_time
                        if fraction_of_time <= 1:
                            x = current_task['location'][0] + fraction_of_time * (
                                        next_task['location'][0] - current_task['location'][0])
                            y = current_task['location'][1] + fraction_of_time * (
                                        next_task['location'][1] - current_task['location'][1])
                            agent['trajectory'].append(np.hstack([x, y, angle]))
                        else:
                            agent['trajectory'].append(np.hstack([next_task['location'][0], next_task['location'][1], angle]))
                    else:
                        agent['trajectory'].append(np.array([next_task['location'][0], next_task['location'][1], angle]))
            while time_step < self.current_time:
                time_step += self.dt
                agent['trajectory'].append(np.array([self.depot_dic[agent['species']]['location'][0], self.depot_dic[agent['species']]['location'][1], angle]))

    def get_episode_reward(self, max_time=100):
        self.calculate_waiting_time()
        eff = self.get_efficiency()
        finished_tasks = self.get_matrix(self.task_dic, 'finished')
        dist = np.sum(self.get_matrix(self.agent_dic, 'travel_dist'))
        reward = - self.current_time - eff * 10 if self.finished else - max_time - eff * 10
        return reward, finished_tasks

    def get_efficiency(self):
        for task in self.task_dic.values():
            if task['feasible_assignment']:
                task['efficiency'] = abs(np.sum(task['requirements'] - task['status'])) / task['requirements'].sum()
            else:
                task['efficiency'] = 10
        efficiency = np.mean(self.get_matrix(self.task_dic, 'efficiency'))
        return efficiency

    def stack_trajectory(self):
        for agent in self.agent_dic.values():
            agent['trajectory'] = np.vstack(agent['trajectory'])

    def plot_animation(self, path, n):
        self.generate_traj()
        plot_robot_icon = False
        if plot_robot_icon:
            drone = plt.imread('env/drone.png')
            drone_oi = OffsetImage(drone, zoom=0.05)

        def get_cmap(n, name='Dark2'):
            '''
            Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
            RGB color; the keyword argument name must be a standard mpl colormap name.
            '''
            return plt.cm.get_cmap(name, n)

        cmap = get_cmap(self.species_num)
        # Set up the plot
        self.stack_trajectory()
        finished_tasks = self.get_matrix(self.task_dic, 'finished')
        finished_rate = np.sum(finished_tasks) / len(finished_tasks)
        gif_len = int(self.current_time/self.dt)
        fig, ax = plt.subplots(dpi=100)
        ax.set_xlim(-0.5, 10.5)
        ax.set_ylim(-0.5, 10.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        plt.subplots_adjust(left=0, right=0.85, top=0.87, bottom=0.02)
        lines = [ax.plot([], [], color=cmap(a['species']), zorder=0)[0] for a in self.agent_dic.values()]
        ax.set_title(f'Agents finish {finished_rate * 100}% tasks within {self.current_time:.2f}min.'
                     f'\nCurrent time is {0:.2f}min')
        color_map = []
        for i in range(self.species_num):
            color_map.append(patches.Patch(color=cmap(i), label='Agent species ' + str(i)))
        color_map.append(patches.Patch(color='g', label='Finished task'))
        color_map.append(patches.Patch(color='b', label='Unfinished task'))
        # red_patch = patches.Patch(color='r', label='Single agent')
        # yellow_patch = patches.Patch(color='y', label='Two agents')
        # cyan_patch = patches.Patch(color='c', label='Three agents')
        # magenta_patch = patches.Patch(color='m', label='>= Four agents')
        if plot_robot_icon:
            ax.legend(handles=color_map, bbox_to_anchor=(0.99, 0.7))
        else:
            ax.legend(handles=color_map,
                      bbox_to_anchor=(0.99, 0.7))
        task_squares = [ax.add_patch(patches.RegularPolygon(xy=(task['location'][0] * 10,
                                                             task['location'][1] * 10),
                                                            numVertices=int(task['requirements'].sum()) + 3,
                                                            radius=0.3, color='b')) for task in self.task_dic.values()]
        depot_tri = [ax.add_patch(patches.Circle((depot['location'][0] * 10,
                                                  depot['location'][1] * 10),
                                                 0.2, color='r')) for depot in self.depot_dic.values()]
        agent_group = [ax.text(agent['location'][0] * 10, agent['location'][1] * 10, str(agent['ID']),
                               horizontalalignment='center', verticalalignment='center', fontsize=8) for agent in self.agent_dic.values()]
        if plot_robot_icon:
            agent_triangles = []
            for a in self.agent_dic.values():
                agent_triangles.append(ax.add_artist(AnnotationBbox(drone_oi, (self.depot_dic[a['species']]['location'][0] * 10,
                                                     self.depot_dic[a['species']]['location'][1] * 10),
                                       frameon=False)))
        else:
            agent_triangles = [ax.add_patch(patches.RegularPolygon(xy=(self.depot_dic[a['species']]['location'][0] * 10,
                                                                       self.depot_dic[a['species']]['location'][1] * 10), numVertices=3,
                                                                   radius=0.2, color=cmap(a['species'])))
                               for a in self.agent_dic.values()]

        # Define the update function for the animation
        def update(frame):
            ax.set_title(f'Agents finish {finished_rate * 100}% tasks within {self.current_time:.2f}min.'
                         f'\nCurrent time is {frame * self.dt:.2f}min')
            pos = np.round([agent['trajectory'][frame, 0:2] for agent in self.agent_dic.values()], 4)
            unq, count = np.unique(pos, axis=0, return_counts=True)
            for agent in self.agent_dic.values():
                repeats = int(count[np.argwhere(np.all(unq == np.round(agent['trajectory'][frame, 0:2], 4), axis=1))])
                agent_triangles[agent['ID']].xy = tuple(agent['trajectory'][frame, 0:2] * 10)
                agent_group[agent['ID']].set_position(tuple(agent['trajectory'][frame, 0:2] * 10))
                agent_group[agent['ID']].set_text(str(repeats))
                if plot_robot_icon:
                    agent_triangles[agent['ID']].xyann = tuple(agent['trajectory'][frame, 0:2] * 10)
                    agent_triangles[agent['ID']].xybox = tuple(agent['trajectory'][frame, 0:2] * 10)
                # else:
                #     agent_triangles[agent['ID']].set_color('m' if repeats >= 4 else 'c' if repeats == 3
                #                                            else 'y' if repeats == 2 else 'r')
                agent_triangles[agent['ID']].orientation = agent['trajectory'][frame, 2] - np.pi / 2
                # Add the current frame's data point to the plot for each trajectory
                if frame > 40:
                    lines[agent['ID']].set_data(agent['trajectory'][frame - 40:frame + 1, 0] * 10,
                                                agent['trajectory'][frame - 40:frame + 1, 1] * 10)
                else:
                    lines[agent['ID']].set_data(agent['trajectory'][:frame + 1, 0] * 10,
                                                agent['trajectory'][:frame + 1, 1] * 10)

            for task in self.task_dic.values():
                if self.reactive_planning:
                    if task['ID'] > np.clip(frame * self.dt//10 * 20 + 20, 20, 100):
                        task_squares[task['ID']].set_color('w')
                        task_squares[task['ID']].set_zorder(0)
                    else:
                        task_squares[task['ID']].set_color('b')
                        task_squares[task['ID']].set_zorder(1)
                if frame * self.dt >= task['time_finish'] > 0:
                    task_squares[task['ID']].set_color('g')
            return lines

        # Set up the animation
        ani = FuncAnimation(fig, update, frames=gif_len, interval=100, blit=True)
        ani.save(f'{path}/episode_{n}_{self.current_time:.1f}.gif')

    def execute_by_route(self, path='./', method=0, plot_figure=False):
        self.plot_figure = plot_figure
        self.max_waiting_time = 200
        while not self.finished and self.current_time < 200:
            decision_agents, current_time = self.next_decision()
            self.current_time = current_time
            decision_agents = decision_agents[0] + decision_agents[1]
            for agent in decision_agents:
                if self.agent_dic[agent]['pre_set_route'] is None or not self.agent_dic[agent]['pre_set_route']:
                    self.agent_step(agent, 0, 0)
                    self.agent_dic[agent]['next_decision'] = np.nan
                    continue
                self.agent_step(agent, self.agent_dic[agent]['pre_set_route'].pop(0), 0)
            self.finished = self.check_finished()
        if self.plot_figure:
            self.plot_animation(path, method)
        print(self.current_time)
        return self.current_time

    def execute_greedy_action(self, path='./', method=0, plot_figure=False):
        self.plot_figure = plot_figure
        while not self.finished and self.current_time < 200:
            release_agents, current_time = self.next_decision()
            self.current_time = current_time
            while release_agents[0] or release_agents[1]:
                agent_id = release_agents[0].pop(0) if release_agents[0] else release_agents[1].pop(0)
                agent = self.agent_dic[agent_id]
                tasks_info, agents_info, mask = self.agent_observe(agent_id, max_waiting=True)
                dist = np.inf
                action = None
                for task_id, masked in enumerate(mask[0, :]):
                    if not masked:
                        dist_ = self.calculate_eulidean_distance(agent, self.task_dic[
                            task_id - 1]) if task_id - 1 >= 0 else self.calculate_eulidean_distance(agent,
                                                                                          self.depot_dic[agent['species']])
                        if dist_ < dist:
                            action = task_id
                self.agent_step(agent_id, action, 0)
            self.finished = self.check_finished()
        if self.plot_figure:
            self.plot_animation(path, method)
        print(self.current_time)
        return self.current_time

    def pre_set_route(self, routes, agent_id):
        if not self.agent_dic[agent_id]['pre_set_route']:
            self.agent_dic[agent_id]['pre_set_route'] = routes
        else:
            self.agent_dic[agent_id]['pre_set_route'] += routes

    def process_map(self, path):
        import pandas as pd
        grouped_tasks = dict()
        groups = list(set(np.array(self.get_matrix(self.task_dic, 'requirements')).squeeze(1).tolist()))
        for task_requirement in groups:
            grouped_tasks[task_requirement] = dict()
        index = np.zeros_like(groups)
        for i, task in self.task_dic.items():
            requirement = int(task['requirements'])
            ind = index[groups.index(requirement)]
            grouped_tasks[requirement].update({ind: task})
            index[groups.index(requirement)] += 1
        grouped_tasks = {key: value for key, value in grouped_tasks.items() if len(value) > 0}
        time_finished = [self.get_matrix(dic, 'time_finish') for dic in grouped_tasks.values()]
        t = 0
        time_tick_stamp = dict()
        while t <= self.current_time:
            time_tick_stamp[t] = [np.sum(np.array(ratio) < t)/len(ratio) for ratio in time_finished]
            t += 0.1
            t = np.round(t, 1)
        pd = pd.DataFrame(time_tick_stamp)
        pd.to_csv(f'{path}time_RL.csv')


if __name__ == '__main__':
    import pickle
    testSet = 'RALTestSet'
    os.mkdir(f'../{testSet}')
    for i in range(50):
        env = TaskEnv((3, 3), (5, 5), (20, 20), 5, seed=i)
        pickle.dump(env, open(f'../{testSet}/env_{i}.pkl', 'wb'))
    env.init_state()
