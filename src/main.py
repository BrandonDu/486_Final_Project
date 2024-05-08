import numpy as np
import matplotlib.pyplot as plt
from spn.algorithms.MPE import mpe

np.random.seed(42)


def generate_data(mu_x, std_x, mu_y, std_y, num_points, label):
    X = np.random.normal(mu_x, std_x, num_points)
    Y = np.random.normal(mu_y, std_y, num_points)
    return np.column_stack((X, Y, np.full(num_points, label)))


if __name__ == "__main__":
    from spn.algorithms.LearningWrappers import learn_parametric, learn_classifier
    from spn.structure.leaves.parametric.Parametric import Categorical, Gaussian
    from spn.structure.Base import Context

    num_points = 300


    # clusters = [
    #     {"mean_x": 1, "std_x": 0.6, "mean_y": 4, "std_y": 1.6, "label": 0},  # Cluster 0
    #     {"mean_x": -3, "std_x": 0.8, "mean_y": -4, "std_y": 1.7, "label": 1},  # Cluster 1
    #     {"mean_x": 5, "std_x": 0.6, "mean_y": -8, "std_y": 1.6, "label": 2},  # Cluster 2
    #     {"mean_x": 8, "std_x": 0.5, "mean_y": 9, "std_y": 1, "label": 3},  # Cluster 3
    #     {"mean_x": -10, "std_x": 0.7, "mean_y": 1, "std_y": 1.4, "label": 4},  # Cluster 4
    # ]
    # colors = ["Red", "Blue", "Green", "Orange", "Purple"]

    clusters = [
        {"mean_x": -1.25, "std_x": 0.65, "mean_y": 0, "std_y": 1.1, "label": 0},  # Cluster 0
        {"mean_x": 1.25, "std_x": 0.65, "mean_y": 0, "std_y": 1.1, "label": 1},  # Cluster 1
    ]

    colors = ["Red", "Blue"]
    num_clusters = len(clusters)

    data = np.array(
        [
            generate_data(
                cluster["mean_x"],
                cluster["std_x"],
                cluster["mean_y"],
                cluster["std_y"],
                num_points,
                clusters.index(cluster),
            )
            for cluster in clusters
        ]
    )
    train_data = data.reshape(num_points * num_clusters, 3)

    for cluster, color in zip(clusters, colors):
        points = data[clusters.index(cluster)]
        plt.scatter(points[:, 0], points[:, 1], color=color, label=f"Cluster {cluster['label']}", s=2)

    # plt.scatter(-1.25, 0, color='Red', edgecolors='Black', s=50, linewidth=1.5)
    # plt.scatter(1.25, 0, color='Blue', edgecolors='Black', s=50, linewidth=1.5)
    plt.title("Gaussian Clusters in the Euclidean Plane")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    # plt.legend()
    # plt.show()
    spn_classification = learn_classifier(
        train_data,
        Context(parametric_types=[Gaussian, Gaussian, Categorical]).add_domains(train_data),
        learn_parametric,
        2,
    )
    #Defined Clusters

    # test_clusters = [
    #     {"mean_x": 1, "std_x": 0.6, "mean_y": 4, "std_y": 1.6, "label": 0},
    #     {"mean_x": -3, "std_x": 0.8, "mean_y": -4, "std_y": 1.7, "label": 1},
    #     {"mean_x": 5, "std_x": 0.6, "mean_y": -8, "std_y": 1.6, "label": 2},
    #     {"mean_x": 8, "std_x": 0.5, "mean_y": 9, "std_y": 1, "label": 3},
    #     {"mean_x": -10, "std_x": 0.7, "mean_y": 1, "std_y": 1.4, "label": 4},
    # ]
    #
    # num_test_points = 10
    # test_data = np.array(
    #     [
    #         generate_data(
    #             cluster["mean_x"],
    #             cluster["std_x"],
    #             cluster["mean_y"],
    #             cluster["std_y"],
    #             num_test_points,
    #             clusters.index(cluster),
    #         )
    #         for cluster in clusters
    #     ]
    # ).reshape(num_clusters * num_test_points, 3)

    # Grid
    x_range = np.arange(-4, 4, 0.1)
    y_range = np.arange(-4, 4, 0.1)
    x_grid, y_grid = np.meshgrid(x_range, y_range)
    points = np.stack([x_grid.flatten(), y_grid.flatten()], axis=-1)
    num_test_points = points.shape[0]
    test_data = np.hstack([points, np.full((num_test_points, 1), np.nan)])
    test_classification = np.copy(test_data)
    test_classification[:, -1] = np.nan

    # test_clusters = [
    #         {"mean_x": -1, "std_x": 1, "mean_y": 1, "std_y": 0.9, "label": 0},
    #         {"mean_x": 1, "std_x": 1.2, "mean_y": 1, "std_y": 1.4, "label": 1},
    #
    #     ]


    classification = mpe(spn_classification, test_classification)
    # correct_classification = 0
    # for i in range(num_test_points):
    #     if test_data[i, -1] == classification[i, -1]:
    #         correct_classification += 1
    classified_clusters = []
    for cluster in range(num_clusters):
        classified_clusters.append(classification[classification[:, -1] == cluster])

    print(classified_clusters)
    for cluster, color in zip(range(num_clusters), colors):
        points = classified_clusters[cluster]
        plt.scatter(points[:, 0], points[:, 1], color=color, label=f"Classification Cluster {cluster}",
                    s=5, alpha=0.3)
    # plt.legend()
    plt.show()
    from spn.io.Graphics import plot_spn
    plot_spn(spn_classification, 'spn.png')

    from spn.io.Text import spn_to_str_equation
    txt = spn_to_str_equation(spn_classification)
    print(txt)

    # print(correct_classification / num_test_points)
