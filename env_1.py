import numpy as np
import math
from gym import spaces
import random
import csv
random.seed(10)

class Env(object):
    def __init__(self, env_params):
        self.step_limit = env_params['step_limit']
        self.days = env_params['days'] # how many days in all
        self.line_days = env_params['line_days'] # how many days for a line to sell
        self.hours = env_params['hours'] # how many hours to deal in a day
        self.time_interval = env_params['time_interval'] # time interval to adjust price
        self.max_action = env_params['max_action']
        self.min_action = env_params['min_action']
        self.max_util = env_params['max_util']
        self.min_util = env_params['min_util']
        self.max_rt = env_params['max_rt']
        self.min_rt = env_params['min_rt']
        self.max_price = env_params['max_price']
        self.min_price = env_params['min_price']
        self.low_state = np.array([self.min_util, self.max_rt, self.min_price])
        self.high_state = np.array([self.max_util, self.min_rt, self.max_price])
        self.total_inventory = env_params['total_inventory'] # total inventory on a ship
        self.customers = env_params['customers']  # customer model
        self.expected_price_mean = env_params['expected_price_mean']  # model of expected price of customer
        self.expected_price_var = env_params['expected_price_var']  # model of expected price of customer
        self.init_price = env_params['init_price']
        self.current_inventory = env_params['total_inventory'] # remaining inventory of a ship
        self.tmp_inventory = self.current_inventory
        self.current_days = 0
        self.current_line_days = 0
        self.current_hours = 0
        self.ct = env_params['ct']
        #self.low_ct =

        self.action_space = spaces.Box(
            low=self.min_action,
            high=self.max_action,
            shape=(1,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=self.low_state,
            high=self.high_state,
            dtype=np.float32
        )
        self.ct_dim = (128,1)

        self.set_s(0,self.max_rt, self.init_price, self.ct[self.current_days])
        self.copy_params()
        # making a copy for current state
    def copy_params(self):
        self.copy_customers = self.customers
        self.copy_expected_price_mean = self.expected_price_mean  # model of expected price of customer
        self.copy_expected_price_var = self.expected_price_var  # model of expected price of customer
        self.copy_init_price = self.init_price
        self.copy_current_inventory = self.current_inventory  # remaining inventory of a ship
        self.copy_current_days = self.current_days
        self.copy_current_line_days = self.current_line_days
        self.copy_current_hours = self.current_hours
        self.copy_ct = self.ct
        self.copy_p = self.p
        self.copy_util = self.util
        self.copy_rt = self.rt
        self.copy_C = self.C
        self.copy_state = self.state


    def load_params(self):
        self.customers = self.copy_customers
        self.expected_price_mean = self.copy_expected_price_mean  # model of expected price of customer
        self.expected_price_var = self.copy_expected_price_var  # model of expected price of customer
        self.init_price = self.copy_init_price
        self.current_inventory = self.copy_current_inventory  # remaining inventory of a ship
        self.current_days = self.copy_current_days
        self.current_line_days = self.copy_current_line_days
        self.current_hours = self.copy_current_hours
        self.ct = self.copy_ct
        self.p = self.copy_p
        self.util = self.copy_util
        self.rt = self.copy_rt
        self.C = self.copy_C
        self.state = self.copy_state

    def set_c(self, ct):
        self.ct = ct
        self.C = self.ct[self.current_days]

    # opening price
    def set_s(self, util, rt, p, C):
        self.p = p
        self.util = util
        self.rt = rt
        self.C = C
        self.state = np.array([self.util, self.rt, self.p])

    def update_p(self, action):
        self.p = self.p*(1 + action/100)

    # state transition after action
    def step(self, action):
        self.tmp_inventory = self.total_inventory - self.current_inventory #记录此次step前的销量
        # when (remaining_time=0 or utilization=100%): done=1
        sum_inventory = 0
        done = 0
        self.update_p(action)
        #计算有多少用户在这一时刻要买,以航次为周期
        num_customers = math.ceil(self.customers[self.current_line_days * (int(self.hours / self.time_interval)) + int(self.current_hours / self.time_interval)])
        #每个用户对应的心理预期价格分布
        #print('------HERE IS self.current_days------')
        #print(self.current_days)
        #print(self.days)
        #print('-------------------------------------')
        expected_price = np.random.normal(self.expected_price_mean[self.current_days], self.expected_price_var, num_customers)
        #print(expected_price)
        #print("self.p:", self.p)
        for i in range(num_customers):
            if expected_price[i] >= self.p:
                self.current_inventory -= 1
                sum_inventory += 1
        #print(self.tmp_inventory)
        reward = self.p * sum_inventory / math.exp(self.tmp_inventory / self.total_inventory)/ self.time_interval
        #reward = self.p * sum_inventory / (self.tmp_inventory + sum_inventory)/ self.time_interval
        revenue = self.p * sum_inventory

        self.current_hours += self.time_interval
        self.rt -= self.time_interval
        if self.current_hours >= self.hours: # 如果是当天最后一笔订单
            self.current_days += 1 # 进入下一天
            self.current_line_days += 1
            self.current_hours = 0
            self.rt -= 24 - self.hours
            if self.current_days < self.days:
                #print("---------current days---------")
                #print(len(self.ct))
                #print(self.current_days)
                self.C = self.ct[self.current_days]

        self.util = (1 - self.current_inventory/self.total_inventory)*100


        if self.rt <= self.min_rt or self.util >= self.max_util or self.current_line_days >= self.line_days:
            self.line_reset()

        next_state = np.array([self.util, self.rt, self.p])
        self.state = next_state

        if self.current_days >= self.days:
            done = 1
        return next_state, self.C, reward, done, {}, self.p, revenue

    # when the remainig time is 0 or the util is 100%, then reset the line
    def line_reset(self):
        self.util = self.min_util
        self.rt = self.max_rt
        self.current_line_days = 0
        self.current_hours = 0
        self.current_inventory = self.total_inventory

    def reset(self):
        self.current_inventory = self.total_inventory  # remaining inventory of a ship
        self.current_days = 0
        self.current_line_days = 0
        self.current_hours = 0
        self.set_s(0, self.max_rt, self.init_price, self.ct[self.current_days])
        self.C = self.ct[self.current_days]
        self.state = np.array([self.util, self.rt, self.p])
        return np.array(self.state),self.C

    def limit(self):
        return self.step_limit

    def generate(self, steps):
        price = []
        step = steps - 5
        for s in range(step):
            # 模拟一次选择
            if (s+1) % 5 == 0:
                price.append(self.p)
            self.step((1000 + 10 * self.util - self.p)/self.p * 100)

        price.append(self.p)
        return np.array(price)



    # TODO change environment setting
    # reset Env to the initial state
    # if is_random=1, randomly set initial current_inventory, current_day and current_interval
    # if set_dis=1, set the distribution using parameters
    # otherwise, use default setting
    '''def yaxin_reset(self, is_random=0, lstm_p=0.5, set_dis=0, poisson_lam=4, normal_mean=0.5, normal_var=0.1):
        # default: remaining_time=1 [=days], utilization=0%, price=0.5 [=1.5*lowest_price]
        # return [remaining_time, utilization, price]
        if is_random:
            self.current_inventory = np.random.randint(0, self.total_inventory)
            self.current_day = np.random.randint(0, self.days)
            self.current_interval = np.random.randint(0, self.num_interval)
            self.p = 0.3 + 0.4 * np.random.random()
        else:
            self.current_inventory = self.total_inventory
            self.current_day = self.days - 1
            self.current_interval = 0
            self.p = lstm_p
        if set_dis:
            self.customers = \
                np.random.poisson(poisson_lam, self.days * self.num_interval)
            self.expected_price_mean = normal_mean
            self.expected_price_var = normal_var
        else:
            self.customers = \
                np.random.poisson(self.total_inventory/self.days/self.num_interval, self.days * self.num_interval)
            self.expected_price_mean = 0.5
            self.expected_price_var = 0.1
        # the number of customers at current_interval of current_day:
        # self.customers[self.current_day*self.num_interval + self.num_interval - self.current_interval]
        remaining_time = (self.current_day+1)/self.days
        # remaining_time = self.current_day/self.days*(1+(self.num_interval - self.current_interval)/self.num_interval)
        utilization = 1 - self.current_inventory/self.total_inventory
        return np.array([[remaining_time, utilization, self.p]])'''
