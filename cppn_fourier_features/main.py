# using some code from: https://github.com/tancik/fourier-feature-networks/blob/master/Demo.ipynb

#%% 
import imageio.v2 as imageio
import torch
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
from tqdm.notebook import trange

from cppn_torch import ImageCPPN, CPPNConfig
from cppn_torch.activation_functions import *
from cppn_torch.fitness_functions import *

from sgd_weights import sgd_weights

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(42);

# Function definitions
#%%
# utils
def save_image(img, filename):
    plt.imshow(img)
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()

#%%
# Fourier feature mapping
def input_mapping(X, B):
    # from original paper
  if B is None:
    return X
  else:
    X_proj = (2.*torch.pi*X) @ B.T
    f0 = torch.sin(X_proj)
    f1 = torch.cos(X_proj)
    return torch.cat([f0, f1], dim=-1)

#%%
# Loss function
def loss(imgs, target):
    loss = 0.5 * torch.mean((imgs - target) ** 2, dim=(1, 2, 3))
    # psnr = -10 * torch.log10(2.*loss)
    return loss
    
    
    # imgs = imgs.permute(0, 3, 1, 2) 
    # target = target.permute(0, 3, 1, 2).repeat(imgs.shape[0], 1, 1, 1)
    # return 1.0-psnr(imgs, target)
    # return torch.mean((targets - output)**2, dim=(1,2,3))


#%%
# Params
n_features = 64
B_scale = 10.0 # 10.0 worked best in original paper
img_res = 32
gens = 100
pop_size = 10
t_size = 3
t_winners = 1
elitism = 1


# Configure CPPN
config = CPPNConfig()
config.activations = [tanh, sigmoid, relu, gauss, identity, sin]
# config.activations = [tanh]
config.res_h = img_res
config.res_w = img_res
config.sgd_steps = 40
config.sgd_learning_rate = .05
config.prob_mutate_weight = 0.0 # no weight mutation
config.num_inputs = n_features + (2 + config.use_radial_distance + config.use_input_bias)
config.hidden_nodes_at_start = 0
# config.normalize_outputs = False

#%%
# Load target image

# image_url = 'https://live.staticflickr.com/7492/15677707699_d9d67acf9d_b.jpg'
image_url = 'https://upload.wikimedia.org/wikipedia/commons/1/1e/Sunrise_over_the_sea.jpg'
# image_url = 'https://i.pinimg.com/originals/65/32/71/653271d0daf4ac594ee6d60b05af9b7c.jpg'

target = imageio.imread(image_url)[..., :3] / 255.
c = [target.shape[0]//2, target.shape[1]//2]
r = img_res//2
target = target[c[0]-r:c[0]+r, c[1]-r:c[1]+r]

plt.title(f'Target Image {target.shape}')
plt.imshow(target)
plt.close();

# convert to torch tensor:
target = torch.from_numpy(target).float().unsqueeze(0).to(device)

#%%
# Prepare inputs
num_coord_inputs = config.num_inputs - n_features

const_inputs = ImageCPPN.initialize_inputs(
    config.res_h, 
    config.res_w,
    config.use_radial_distance,
    config.use_input_bias,
    num_coord_inputs,
    device=device,
    coord_range=(-0.5, 0.5)
    )

B = torch.randn((n_features//2, num_coord_inputs), device=device) * B_scale
X = input_mapping(const_inputs, B)
X = torch.cat([const_inputs, X], dim=-1)

print("inputs shape:", X.shape)

#%%
# Tournament selection
def tournament_selection(population):
    new_pop = population[:elitism] # elitism
    while len(new_pop) < pop_size:
        random_inds = torch.randint(0, len(population), (t_size,))
        subpop = [population[i] for i in random_inds]
        subpop = sorted(subpop, key=lambda x: x.fitness, reverse=True)
        winners = subpop[:t_winners]
        new_pop.extend(winners)
        
    new_pop = sorted(new_pop, key=lambda x: x.fitness, reverse=True)
    return new_pop

#%%
# Evolve
population = []
for i in range(pop_size):
    population.append(ImageCPPN(config))
    # population[i].mutate()

fits = 1.0-loss(torch.stack([ind(X) for ind in population]), target)
for i, fit in enumerate(fits):
    population[i].fitness = fit

pbar = trange(gens)
try:
    for gen in pbar:
        children = []
        for i in range(len(population)):
            child = population[i].clone(new_id=True)
            child.mutate()
            children.append(child)
        
        sgd_weights(children, X, target, loss, config)
            
        imgs = [child(X) for child in children]
        
        fits = 1.0-loss(torch.stack(imgs), target)
        for i, fit in enumerate(fits):
            children[i].fitness = fit
        
        population = tournament_selection(population + children)

        save_image(population[0](X).detach().cpu(), f'current_best.png')
        pbar.set_description(f'f:{population[0].fitness.item():.4f}')
        
except KeyboardInterrupt:
    print("ending early")
    
# %%
# Show result
img = population[0](X).detach().cpu()
plt.imshow(img)
plt.show();



# %%
