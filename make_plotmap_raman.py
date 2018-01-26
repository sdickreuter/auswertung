plt.figure(figsize=(5, 5))
plt.imshow(data)
for x, y, s in zip(xy[:, 0], xy[:, 1], np.round(ram, 1)):
    plt.text(x - 1, y - 5, str(s))

plt.savefig(savedir + "grid_max.pdf", format='pdf')
plt.close()