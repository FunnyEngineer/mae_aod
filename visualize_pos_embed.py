from util.pos_embed import get_2d_sincos_pos_embed
import pdb
import matplotlib.pyplot as plt

test = get_2d_sincos_pos_embed(2, 20, cls_token=False)
test = test.reshape(20, 20, 2)
fig, axs = plt.subplots(1, 4)
for i, ax in enumerate(axs):
    ax.imshow(test[:, :, i])
plt.show()
pdb.set_trace()