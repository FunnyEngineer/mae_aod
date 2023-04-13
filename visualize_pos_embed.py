from util.pos_embed import get_2d_sincos_pos_embed
import pdb
import matplotlib.pyplot as plt

test = get_2d_sincos_pos_embed(4, 20, cls_token=False)
test = test.reshape(20, 20, 4)
fig, axs = plt.subplots(2, 2)

i = 0
for col in range(2):
    for row in range(2):
        pcm = axs[col, row].imshow(test[:, :, i])
        fig.colorbar(pcm, ax=axs[col, row])
        i += 1

plt.show()
pdb.set_trace()