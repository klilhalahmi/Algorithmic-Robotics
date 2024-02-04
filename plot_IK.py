import numpy as np
import matplotlib.pyplot as plt

print(trajectory)

if __name__ == "__main__":
    xline = []
    yline = []
    zline = []

    for i in range(count):
        xline.append(trajectory[i][0])
        yline.append(trajectory[i][1])
        zline.append(trajectory[i][2])

    xline = np.concatenate(xline)
    yline = np.concatenate(yline)
    zline = np.concatenate(zline)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(xline, yline, zline)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    ax.set_xlim([0, 0.1])
    ax.set_ylim([0.367, 0.567])
    ax.set_zlim([0.418, 0.718])

    plt.show()
