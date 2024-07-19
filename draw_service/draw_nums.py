import csv
from matplotlib import pyplot as plt


def draw_bar():
    with open("nums.txt", 'r') as f:
        csv_reader = csv.reader(f)
        target_data=[]
        for row in csv_reader:
            row=[int(i) for i in row]
            target_data.append(row)

    x=list(range(0,len(target_data[1])))

    plt.bar(x, target_data[1])
    plt.show()


if __name__ == '__main__':
    draw_bar()