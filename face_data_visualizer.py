import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from sklearn.datasets import make_blobs


class FaceDataVisualizer:
    def __init__(self, ax1, ax2, data):
        self.ax1 = ax1
        self.ax2 = ax2
        self.data = data
        self.collections = ax1.scatter(data[:, 1], data[:, 2], s=10, picker=5)
        self.annot = ax1.annotate("", xy=(0, 0), xytext=(-20, 20), textcoords="offset points",
                                  bbox=dict(boxstyle="round", fc="w"),
                                  arrowprops=dict(arrowstyle="-"), horizontalalignment='center')
        self.background = self.ax1.figure.canvas.copy_from_bbox(self.ax1.bbox)

        self.annot.set_visible(False)
        self.ax2.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
        self.ax2.set_xlabel('selected frame_id: ')
        self.ax2.legend(ncol=4, bbox_to_anchor=(0, 1),
                        loc='lower left', fontsize='small')

    def on_pick(self, event):
        if event.artist != self.collections:
            return True

        N = len(event.ind)
        if not N:
            return True

        self.update_img(event.ind[0])

    def update_img(self, ind):
        img = mpimg.imread('./img/face' + str(ind % 2) + '.png')
        self.ax2.imshow(img)
        self.ax2.set_xlabel('selected frame id: ' + str(ind))
        self.ax2.figure.canvas.draw_idle()

    def on_hover(self, event):
        vis = self.annot.get_visible()
        if event.inaxes == self.ax1:
            cont, ind = self.collections.contains(event)
            if cont:
                self.update_annot(ind)
                self.annot.set_visible(True)
                self.ax1.figure.canvas.restore_region(self.background)
                self.ax1.draw_artist(self.annot)
                self.ax1.figure.canvas.blit(self.ax1.bbox)
            else:
                if vis:
                    self.annot.set_visible(False)
                    self.ax1.figure.canvas.restore_region(self.background)
                    self.ax1.draw_artist(self.annot)
                    self.ax1.figure.canvas.blit(self.ax1.bbox)

    def update_annot(self, ind):
        pos = self.collections.get_offsets()[ind["ind"][0]]
        self.annot.xy = pos
        text = "frame_id: {:d}\nx: {:.3f}\ny: {:.3f}".format(ind["ind"][0] + 1, pos[0], pos[1])
        self.annot.set_text(text)


if __name__ == "__main__":
    n_rows = 10000
    n_cols = 700
    outliers_fraction = 0.01
    n_outliers = int(outliers_fraction * n_rows)
    n_inliers = n_rows - n_outliers
    rng = np.random.RandomState(42)

    data_in_pos = make_blobs(centers=[[0, 0]], cluster_std=0.5, random_state=0,
                             n_samples=n_inliers, n_features=2)[0]
    data_out_pos = rng.uniform(low=-3, high=3, size=(n_outliers, 2))
    data_cols = rng.uniform(low=-3, high=3, size=(n_rows, n_cols - 1))
    data_first_col = np.arange(1, n_rows + 1).reshape(n_rows, 1)
    data_pos = np.concatenate([data_in_pos, data_out_pos], axis=0)
    data = np.concatenate([data_first_col, data_pos, data_cols], axis=1)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

    fdv = FaceDataVisualizer(ax1, ax2, data)

    fig.canvas.mpl_connect('pick_event', fdv.on_pick)
    fig.canvas.mpl_connect("motion_notify_event", fdv.on_hover)

    plt.show()
