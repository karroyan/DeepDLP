import csv
from env_1 import Env
from ddpg import *
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(10)


EPISODES = 200 #set to 100 to test the code
TEST = 10
STEP_LIMIT = 125

# parameters for env
DAYS = 25
LINE_DAYS = 7
HOURS = 10
TIME_INTERVAL = 2
MAX_ACTION = 100
MIN_ACTION = -100
MAX_UTIL = 100
MIN_UTIL = 0
MAX_RT = 168 # Remaining Hours
MIN_RT = 0
MAX_PRICE = 2300
MIN_PRICE = 600
TOTAL_INVENTORY = 800
'''
EXPECTED_PRICE_MEAN = [1000, 1020, 1050, 1100, 1160,\
                       1240, 1320, 1400, 1510, 1620,\
                       1700, 1740, 1800, 1900, 1950,\
                       1880, 1840, 1810, 1720, 1650,\
                       1590, 1520, 1400, 1320, 1300,\
                       1200, 1120, 1090, 1010, 950
                       ]
'''
EXPECTED_PRICE_MEAN = [1700, 1740, 1800, 1900, 1950,\
                       1880, 1840, 1810, 1720, 1650,\
                       1700, 1740, 1800, 1900, 1950,\
                       1880, 1840, 1810, 1720, 1650, \
                       1700, 1740, 1800, 1900, 1950, \
                       1880, 1840, 1810, 1720, 1650
                       ]
EXPECTED_PRICE_VAR = 900
INIT_PRICE = 1800

#parameters for LSTM
BATCH_SIZE = 1
HIDDEN_UNITS = 128
HIDDEN_UNITS1 = 128
LEARNING_RATE = 0.1
EPOCH = 2000
OPENING_DAYS = 25+1
OPENING_DAYS_1 = 25+1+1
CT = [0 for i in range(DAYS+1)]
REMAINING_DAY = 5

Env_params = {
        'days': 29,
        'line_days' : LINE_DAYS,
        'hours' : HOURS,
        'time_interval': TIME_INTERVAL,
        'max_action' : MAX_ACTION,
        'min_action' : MIN_ACTION,
        'max_util' : MAX_UTIL,
        'min_util' : MIN_UTIL,
        'max_rt' : MAX_RT,
        'min_rt' : MIN_RT,
        'max_price' : MAX_PRICE,
        'min_price' : MIN_PRICE,
        'total_inventory': TOTAL_INVENTORY,
        'customers': np.random.poisson(60, int(MAX_RT/TIME_INTERVAL)),
        'expected_price_mean': EXPECTED_PRICE_MEAN,
        'expected_price_var': EXPECTED_PRICE_VAR,
        'init_price': INIT_PRICE,
        'step_limit' : STEP_LIMIT,
        'ct' : CT
    }

def main():
    env = Env(Env_params)
    agent = DDPG(env)

    n_steps = 1
    n_features = 1


    state = env.reset()
    np_price = env.generate(OPENING_DAYS * 5) # opening price of 26 days

    #np_price = EXPECTED_PRICE_MEAN[:126]
    graph = tf.Graph()
    with graph.as_default():
        #------------------------------LSTM layer---------------------------
        inputs = tf.placeholder(np.float32, shape = (BATCH_SIZE, DAYS, 1))
        preds = tf.placeholder(np.float32, shape = (BATCH_SIZE,  1))
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(
            num_units = HIDDEN_UNITS,
            name = "LSTM_CELL"
        )
        '''
        lstm_cell2 =  tf.contrib.rnn.BasicLSTMCell(
            num_units = HIDDEN_UNITS1,
            name = "LSTM_CELL2"
        )
        '''
        #multi_lstm = tf.contrib.rnn.MultiRNNCell(cells=[lstm_cell, lstm_cell2])
        #multi_lstm = tf.contrib.rnn.MultiRNNCell(cells=[lstm_cell])

        # 自己初始化state
        # 第一层state
        lstm_layer1_c = tf.zeros(shape=(BATCH_SIZE, HIDDEN_UNITS1))
        lstm_layer1_h = tf.zeros(shape=(BATCH_SIZE, HIDDEN_UNITS1))
        layer1_state = tf.contrib.rnn.LSTMStateTuple(c=lstm_layer1_c, h=lstm_layer1_h)
        '''
        # 第二层state
        lstm_layer2_c = tf.zeros(shape=(BATCH_SIZE, HIDDEN_UNITS))
        lstm_layer2_h = tf.zeros(shape=(BATCH_SIZE, HIDDEN_UNITS))
        layer2_state = tf.contrib.rnn.LSTMStateTuple(c=lstm_layer2_c, h=lstm_layer2_h)
        '''
        #init_state = (layer1_state, layer2_state)
        init_state = (layer1_state)
        print(init_state)

        # 自己展开RNN计算
        outputs = list()  # 用来接收存储每步的结果
        state_list = list()
        state = init_state
        with tf.variable_scope('RNN'):
            for timestep in range(DAYS):
                if timestep > 0:
                    tf.get_variable_scope().reuse_variables()
                # 这里的state保存了每一层 LSTM 的状态
                #(cell_output, state) = multi_lstm(inputs[:, timestep, :], state)
                (cell_output, state) = lstm_cell(inputs[:, timestep, :], state)
                outputs.append(cell_output)
                state_list.append(state)

        #h = outputs[-1]
        h = tf.layers.dense(outputs[-1], 1)

        '''
        init_state = multi_lstm.zero_state(batch_size=BATCH_SIZE, dtype = np.float32)

        output, state = tf.nn.dynamic_rnn(
        cell = multi_lstm,
        inputs = inputs,
        dtype = tf.float32,
        initial_state = init_state
        )
        h = tf.layers.dense(output[:,:,:], 1)
        '''
        #---------------------------------define loss and optimizer-------------
        mse = tf.losses.mean_squared_error(labels = preds, predictions = h)
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss=mse)

        init = tf.global_variables_initializer()

    #-----------------------------define session--------------------------------

    sess = tf.Session(graph = graph)
    #with tf.Session(graph = graph) as sess:
    sess.run(init)
    for epoch in range(1, EPOCH+1):
        train_losses = []
        test_losses = []

        for j in range(1):
            X_test_label = np.array(np_price[:-1])
            #X_max = np.max(X_test_label)
            #X_min = np.min(X_test_label)
            #scalar = X_max - X_min
            #X_test_label = (X_test_label-X_min)/scalar
            _, train_loss, LSTMtuple, output, output_list = sess.run(
                fetches = (optimizer, mse, state_list, h, outputs),
                feed_dict = {
                    inputs : X_test_label.reshape(1,DAYS,1),#用25天闭盘价预测26天开盘价
                    #preds : X_test_label.reshape(1,25,1)
                    #preds : np.array((np_price[-1]-X_min)/scalar).reshape(1,1)
                    preds: np.array(np_price[-1]).reshape(1, 1)
                }
            )
            train_losses.append(train_loss)
            #output = output*scalar + X_min
            #if (epoch % 10 == 0):
                #print(train_loss)
                #print(output)
    CT = []
    for cts in LSTMtuple:
        CT.append(cts.c.reshape(HIDDEN_UNITS1))
    temp_zero = np.zeros(128)

    #----------------------------DDPG sample and train------------------------------------
    f_reward_train = open('reward_without_xt_train_0207_1.csv', 'w', encoding='utf-8', newline="")
    csv_writer_reward_train = csv.writer(f_reward_train)
    csv_writer_reward_train.writerow(['reward'])

    f_revenue_train = open('revenue_without_xt_train_0207_1.csv', 'w', encoding='utf-8', newline="")
    csv_writer_revenue_train = csv.writer(f_revenue_train)
    csv_writer_revenue_train.writerow(['revenue'])

    env.set_c(CT)
    env.days = 25
    for episode in range(EPISODES):
        #CT.append(temp_zero)
        state, ct = env.reset()
        #env.set_c(CT)
        train_total_reward = 0
        train_total_revenue = 0
        for step in range(env.step_limit):
            action = agent.noise_action(state,ct)
            next_state, next_ct, reward,done,_ ,temp_p, temp_revenue= env.step(action)
            agent.perceive(state,action,ct,reward,next_state,next_ct,done)
            state = next_state
            ct = next_ct
            train_total_reward += reward
            train_total_revenue += temp_revenue
            #if step % 125 == 0:

            #    train_total_reward = 0
            if done:
                print('-----this is step-------',step)
                break
        csv_writer_reward_train.writerow([train_total_reward / 25])
        csv_writer_revenue_train.writerow([train_total_revenue / 25])
        agent.train()
        if (episode % 10 == 0):
            print("trianing episode", episode)

        path = "./nnWeight/weight" + str(int(episode)) + ".h5"
        if episode % 10 == 0:
            agent.save_weights(time_step=episode)

    #-----------------DDPG predict and LSTM--------------------------
    #用后面5天预测，预测1000个episode，5天的reward做平均
    #DDPG训练后的state直接作为26天开始的state

    f = open('result_DDPG_0207_without_xt' + str(episode) + '.csv', 'w', encoding='utf-8', newline="")
    csv_writer = csv.writer(f)
    csv_writer.writerow(["EPISODE", "ave_reward"])

    train_state = state
    train_ct = ct
    env.copy_params()

    f_reward_pred = open('reward_without_xt_pred_0207_1.csv', 'w', encoding='utf-8', newline="")
    csv_writer_reward_pred = csv.writer(f_reward_pred)
    csv_writer_reward_pred.writerow(['reward'])

    f_p = open('p_without_xt_pred_0207_1.csv', 'w', encoding='utf-8', newline="")
    csv_writer_p = csv.writer(f_p)
    csv_writer_p.writerow(['p'])

    revenue_pred = open('revenue_without_xt_pred_0207_1.csv', 'w', encoding='utf-8', newline="")
    csv_writer_revenue_pred = csv.writer(revenue_pred)
    csv_writer_revenue_pred.writerow(['revenue'])

    
    for episode in range(EPISODES):
        tmp_CT = CT
        env.load_params()
        state = train_state
        ct = train_ct
        total_reward = 0
        total_revenue = 0
        done = 0
        new_predict_x = np_price[:-1]
        for day in range(REMAINING_DAY):#26-30
            env.days = 26+day
            #-----------------------------------LTSM predict---------------
            #add new prediction data set
            #需要调整窗口

            new_predict_x_reshape = new_predict_x.reshape(1, DAYS, 1)

            #new_predict_x = np.array(new_predict_x, dtype=np.float32)
            (o_state_list, o_h) = sess.run([state_list, h],\
                                           feed_dict={inputs: new_predict_x_reshape[:, :, :].reshape(1, DAYS, 1)})
            state[2] = o_h
            ct = o_state_list[-1].c.reshape(HIDDEN_UNITS)
            tmp_CT.append(ct)
            env.set_c(tmp_CT)

            print("---which day----")
            print(day)
            for j in range(5):
                # env.render()
                action = agent.action(state, ct)  # direct action for test
                #action = 0
                state, ct, reward, done, _ ,temp_p, temp_revenue= env.step(action)
                csv_writer_p.writerow([temp_p])
                total_reward += reward
                total_revenue += temp_revenue
                if done:
                    break
            new_predict_x = new_predict_x[1:]
            new_predict_x = np.append(new_predict_x, state[2])
            if done:
                break


        ave_reward = total_reward / REMAINING_DAY
        ave_revenue = total_revenue / REMAINING_DAY
        print('---this is ave_reward----',ave_reward)
        print('---this is ave_revenue----', ave_revenue)
        csv_writer_reward_pred.writerow([ave_reward])
        csv_writer_revenue_pred.writerow([ave_revenue])
        print('episode: ', episode, 'Evaluation Average Reward:', ave_reward)
        print('episode: ', episode, 'Evaluation Average Revenue:', ave_revenue)
        # X_max_new = np.max(new_predict_x)
        # X_min_new = np.min(new_predict_x)
        # scalar_new = X_max_new - X_min_new
        # new_predict_x = (new_predict_x - X_min_new) / scalar_new

        # print("----this is ct----")
        # print(LSTMtuple)
        # print(type(LSTMtuple))
        # for new_timestep in range(DAYS):
        #    (cell_output, my_state) = lstm_cell(new_predict_x[:, new_timestep, :], my_state)
        # outputs.append(cell_output)
        # print(type(my_state.c))
        # CT = np.append(CT, my_state.c.reshape(HIDDEN_UNITS1))
        # h = tf.layers.dense(outputs[-1], 1)
        # prediction = output.eval(session = sess, feed_dict = {x_data : test})

        # print(h*scalar_new + X_min_new)
        # print(o_state_list)
        # print(o_h*scalar_new + X_min_new)

if __name__ == '__main__':
    main()
