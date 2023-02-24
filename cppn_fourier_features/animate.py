from matplotlib import pyplot as plt


def save_image(img, filename):
    img.clamp_(0, 1)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()


def animate(population, children, anim_images, inputs, steps):
    champ = population[0]
    if champ in children:
        champ_img = champ(inputs).detach().cpu()
        save_image(champ_img, f'images/current_best.png')
        anim_images.append(champ_img)
    else:
        # remove last few images if no new champion
        anim_images = anim_images[:-steps]
        