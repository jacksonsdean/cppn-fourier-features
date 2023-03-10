# using some code from: https://github.com/tancik/fourier-feature-networks/blob/master/Demo.ipynb

#%% 
import imageio.v2 as imageio
import torch
import cv2
import os
import matplotlib.pyplot as plt
from tqdm import trange
import random
from skimage import img_as_ubyte # for imageio gifs

from cppn_torch import ImageCPPN, CPPNConfig
from cppn_torch.graph_util import activate_population
from cppn_torch.activation_functions import *
from cppn_torch.fitness_functions import *
from cppn_torch.util import visualize_network
from sgd_weights import sgd_weights
from animate import animate, save_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)

# Function definitions

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

#%% # Loss function
def loss(imgs, target):
    imgs = imgs.nan_to_num() # TODO
    # MSE:
    loss = torch.mean((imgs - target) ** 2, dim=(1, 2, 3))
    return loss

    # Other fancier losses:
    imgs, target = correct_dims(imgs, target)
    # loss = 1.0 - ssim(imgs, target)
    # loss = 1.0 - fsim(imgs, target)
    loss = 1.0 - msssim(imgs, target)
    return loss

#%% # Params
# Features
n_features = 64 # 256 good, but uses a lot of memory for higher resolutions
img_res = 64
incl_xy = True
make_gif = False

B_scale = 1.25 # 10.0 worked best in original paper, but lower (~1-3) seems to work better here
# B_scale_factor = 10.0
# B_scale = B_scale_factor * img_res / 256.  # this ratio seems to be more robust to different resolutions?

# Evolution
gens = 100
pop_size = 10
tourn_size = 5
tourn_winners = 1
elitism = 1

# CPPNs
config = CPPNConfig()
config.seed = SEED
config.activations = [tanh, sigmoid, relu, gauss, identity, sin]
config.res_h = img_res
config.res_w = img_res
config.prob_mutate_weight = 0.0 # no weight mutation
config.hidden_nodes_at_start = 10
config.use_input_bias = True
config.node_agg = 'sum'
config.activation_mode = 'layer' # layer slightly faster than node.. population not great yet
config.normalize_outputs = True
# config.prob_add_node = .9 # very high to encourage growth?
# config.prob_add_connection = .8 "
# config.init_connection_probability = 1.0 # more sgd
config.num_inputs = n_features + (incl_xy*2 + config.use_radial_distance + config.use_input_bias)

coord_range = (-0.5, 0.5) # picbreeder/MOVE
#coord_range = (0, 1) # FF let networks learn

# SGD
config.sgd_learning_rate = .15 # high seems to work well at least on sunrise
lr_decay_in_gen = 1.0 # not sure if needed
lr_decay_between_gen = 1.0 # not sure if needed
sgd_every = 1 # anything other than 1 doesn't really make sense with this simple of an EA
config.sgd_steps = 30
min_early_stop_delta = -1e-3 # closer to 0 results in less sgd per generation

#%% # Load target image
image_map = {
   'fox':       'https://live.staticflickr.com/7492/15677707699_d9d67acf9d_b.jpg',
   'sunrise':   'https://upload.wikimedia.org/wikipedia/commons/1/1e/Sunrise_over_the_sea.jpg'
}


# target = imageio.imread(image_map['fox'])
target = imageio.imread(image_map['sunrise'])
target = target[..., :3] / 255. # convert to floats and remove alpha channel
c = [target.shape[0]//2, target.shape[1]//2]

if img_res >= target.shape[0]//2:
    # center crop
    r = img_res//2
    target = target[c[0]-r:c[0]+r, c[1]-r:c[1]+r] 
else:
    # center crop to square and then resize
    r = min(target.shape[:2])//2 # radius for square is min dim
    target = target[c[0]-r:c[0]+r, c[1]-r:c[1]+r]
    target = cv2.resize(target, (img_res, img_res))

#%%
# plt.imshow(target)
# plt.show()
#%%

# convert to torch tensor:
target = torch.from_numpy(target).to(device).float()
target = target.unsqueeze(0)
print("target shape:", target.shape)

os.makedirs("images", exist_ok=True)

#%% # Prepare inputs
num_coord_inputs = (2 + config.use_radial_distance + config.use_input_bias)

const_inputs = ImageCPPN.initialize_inputs(
    config.res_h, 
    config.res_w,
    config.use_radial_distance,
    config.use_input_bias,
    num_coord_inputs,
    device=config.device,
    coord_range=coord_range
    )

B = torch.randn((n_features//2, num_coord_inputs), device=device) * B_scale
X = input_mapping(const_inputs, B)

# kind of gross but allows for more configuration:
if incl_xy:
    X = torch.cat([const_inputs[:,:,:2], X], dim=-1)
if config.use_radial_distance:
    X = torch.cat([const_inputs[:,:, 2].unsqueeze(-1), X], dim=-1)
if config.use_input_bias:
    X = torch.cat([const_inputs[:,:,-1].unsqueeze(-1), X], dim=-1)
    
print("inputs shape:", X.shape, "\n")

#%% # Tournament selection
def tournament_selection(population):
    new_pop = population[:elitism] # done in main loop now
    while len(new_pop) < pop_size:
        random_inds = torch.randint(0, len(population), (tourn_size,))
        subpop = [population[i] for i in random_inds]
        subpop = sorted(subpop, key=lambda x: x.fitness, reverse=True)
        winners = subpop[:tourn_winners]
        new_pop.extend(winners)
        
    new_pop = sorted(new_pop, key=lambda x: x.fitness, reverse=True)
    return new_pop


#%% # Evolve
population = []
for i in range(pop_size):
    population.append(ImageCPPN(config))
    population[-1].mutate()
    population[-1].add_node() # start with extra node
    
if config.activation_mode == "population":
    imgs = activate_population(population, config, X)
else:
    imgs = torch.stack([ind(X) for ind in population])
    
fits = 1.0-loss(imgs, target)

for i, fit in enumerate(fits):
    population[i].fitness = fit

if make_gif:
    anim_images = []
else:
    anim_images = None
steps= 0
pbar = trange(gens)
try:
    for gen in pbar:
        # Reproduction
        children = []
        for i in range(len(population)):
            child = population[i].clone(new_id=True)
            child.mutate()
            children.append(child)
        
        children.extend([g.clone() for g in population[:elitism]])
        
        # SGD
        if (gen+1) % sgd_every == 0:
            steps = sgd_weights(children, X, target, loss, config, anim_images, lr_decay_in_gen, min_early_stop_delta=min_early_stop_delta)
            config.sgd_learning_rate *= lr_decay_between_gen
        
        # Evaluation
        if config.activation_mode == "population":
            imgs = activate_population(children, config, X)
        else:
            imgs = torch.stack([child(X) for child in children])
        fits = 1.0-loss(imgs, target)
        for i, fit in enumerate(fits):
            children[i].fitness = fit
        
        # Selection
        population = tournament_selection(population + children) # sorted
        
        animate(population, children, anim_images, X, steps, make_gif)
        
        pbar.set_description(f'f:{population[0].fitness.item():.4f}')
        
except KeyboardInterrupt:
    print("ending early")
except RuntimeError as e:
    print("RuntimeError:", e)
    
# do a cheeky sgd on the final champion's weights
population = sorted(population, key=lambda x: x.fitness, reverse=True)
config.sgd_steps = 400
config.sgd_learning_rate = 1e-3
sgd_weights(population[:1], X, target, loss, config, anim_images, lr_decay_in_gen, min_early_stop_delta=-1e-5)

#%% Show result
img = population[0](X).detach().cpu()
save_image(img, f'images/final.png')

visualize_network(population[0], save_name=f'images/final_genome.png')

# %% Save gif
if make_gif and anim_images:
    print("saving gif...")
    imageio.mimsave('images/evolution.gif', [img_as_ubyte(i) for i in anim_images], fps=24)
    plt.show();

