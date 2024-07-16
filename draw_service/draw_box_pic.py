import matplotlib.pyplot as plt
import numpy as np
import csv
import base_config

target_dir = base_config.TARGET_DATA + "/" + "draw_pic"

if __name__ == '__main__':

    # face number
    face = [5, 10, 15]
    # resolution
    reso = [240, 360, 480, 720, 1080]

    edge_2_edge_cost = [[], [], []]
    edge_2_edge_error = [[], [], []]

    # 5,10,15曲线都有5个横坐标
    # 文件名称类似于time_5_360,time_5_720/time_10_1080/time_15_640这种
    # 中缀为5/10/15的,对应同一条曲线
    for idx, face_num in enumerate(face):

        file_list = [str(face_num) + "_" + str(xi) for xi in reso]

        for file_name in file_list:
            temp_time = []
            f = open(target_dir + "/" + "time_"+f"{file_name}", 'r')

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

    plt.title('Time Cost With or Without MQ', fontsize=12)

    ax1.set_ylabel('Time Sum (seconds)', fontsize=12)

    print(edge_2_edge_cost[0])
    print(edge_2_edge_error[0])

    ax2 = ax1.twinx()
    li0 = ax2.plot(reso, edge_2_edge_cost[0], label='', color='red')
    li1 = ax2.errorbar(reso, edge_2_edge_cost[0], label=f'With MQ DetPhase', yerr=edge_2_edge_error[0], fmt='o',
                       color='red',
                       capsize=2)
    li2 = ax2.plot(reso, edge_2_edge_cost[1], label='', color='blue')
    li3 = ax2.errorbar(reso, edge_2_edge_cost[1], label=f'With MQ DetPhase', yerr=edge_2_edge_error[1], fmt='o',
                       color='blue',
                       capsize=2)
    li4 = ax2.plot(reso, edge_2_edge_cost[2], label='', color='green')
    li5 = ax2.errorbar(reso, edge_2_edge_cost[2], label=f'With MQ DetPhase', yerr=edge_2_edge_error[2], fmt='o',
                       color='green',
                       capsize=2)
    ax2.set_ylabel('Time PerPic (seconds)', fontsize=12)

    fig.legend(loc='upper left', bbox_to_anchor=(0.13, 0.88))
    fig.show()
