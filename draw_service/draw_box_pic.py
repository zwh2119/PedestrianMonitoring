import matplotlib.pyplot as plt
import numpy as np
import csv
import base_config

target_dir = base_config.TARGET_DATA + "/" + "draw_car_text"

if __name__ == '__main__':

    # car number
    car = [10, 13, 17, 20, 23]
    # resolution
    reso = [240, 360, 480, 720, 1080]

    edge_2_edge_cost = [[], [], [], [], []]
    edge_2_edge_error = [[], [], [], [], []]

    # detection_cost = [[], [], []]
    # detection_error = [[], [], []]
    #
    # classification_cost = [[], [], []]
    # classification_error = [[], [], []]

    # 10,13,17,20,23曲线都有5个横坐标
    # 文件名称类似于time_5_360,time_5_720/time_10_1080/time_15_640这种
    # 中缀为10/13/17/20/23的,对应同一条曲线
    for idx, car_num in enumerate(car):

        file_list = [str(car_num) + "_" + str(xi) for xi in reso]

        for file_name in file_list:
            temp_time = []
            f = open(target_dir + "/" + "traffic_" + f"{file_name}", 'r')

            reader = csv.reader(f)

            time_sum = []
            for row in reader:

                time_sum.append(row[0])

            time_sum = np.array(time_sum, dtype=float)

            edge_2_edge_cost[idx].append(np.mean(time_sum))
            edge_2_edge_error[idx].append(np.std(time_sum))

    # 主要是画时延
    fig, ax1 = plt.subplots(figsize=(10, 6))

    plt.xlabel('Resolution', fontsize=12)
    plt.grid(True)

    plt.title('Time Cost With face and resolution', fontsize=12)

    ax1.set_ylabel('Time Sum (seconds)', fontsize=12)

    li0 = ax1.plot(reso, edge_2_edge_cost[0], label='', color='red')
    li1 = ax1.errorbar(reso, edge_2_edge_cost[0], label=f'10 car', yerr=edge_2_edge_error[0], fmt='o',
                       color='red',
                       capsize=2)
    li2 = ax1.plot(reso, edge_2_edge_cost[1], label='', color='orange')
    li3 = ax1.errorbar(reso, edge_2_edge_cost[1], label=f'13 car', yerr=edge_2_edge_error[1], fmt='o',
                       color='orange',
                       capsize=2)
    li4 = ax1.plot(reso, edge_2_edge_cost[2], label='', color='yellow')
    li5 = ax1.errorbar(reso, edge_2_edge_cost[2], label=f'17 car', yerr=edge_2_edge_error[2], fmt='o',
                       color='yellow',
                       capsize=2)

    li6 = ax1.plot(reso, edge_2_edge_cost[3], label='', color='green')
    li7 = ax1.errorbar(reso, edge_2_edge_cost[3], label=f'20 car', yerr=edge_2_edge_error[3], fmt='o',
                       color='green',
                       capsize=2)
    li8 = ax1.plot(reso, edge_2_edge_cost[4], label='', color='blue')
    li9 = ax1.errorbar(reso, edge_2_edge_cost[4], label=f'23 car', yerr=edge_2_edge_error[4], fmt='o',
                       color='blue',
                       capsize=2)


    fig.legend(loc='upper left', bbox_to_anchor=(0.13, 0.88))
    fig.show()
