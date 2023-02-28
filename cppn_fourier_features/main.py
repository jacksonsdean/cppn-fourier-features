# using some code from: https://github.com/tancik/fourier-feature-networks/blob/master/Demo.ipynb

#%% 
import imageio.v2 as imageio
import torch
import cv2
import os
import matplotlib.pyplot as plt
from tqdm import trange
import random
from skimage import img_as_ubyte

from cppn_torch import ImageCPPN, CPPNConfig
from cppn_torch.activation_functions import *
from cppn_torch.fitness_functions import *
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
    # MSE:
    # loss = torch.mean((imgs - target) ** 2, dim=(1, 2, 3))
    
    # Fancier:
    imgs, target = correct_dims(imgs, target)
    imgs = imgs.nan_to_num()
    # loss = 1.0 - ssim(imgs, target)
    loss = 1.0 - msssim(imgs, target)
    return loss

#%% # Params
# Features
n_features = 64
B_scale = 3.0 # 10.0 worked best in original paper, but lower (~2.0) seems to work better for sunrise
img_res = 256
incl_xy = True
make_gif = True

# Evolution
gens = 1000
pop_size = 10
tourn_size = 5
tourn_winners = 2
elitism = 1

# CPPNs
config = CPPNConfig()
config.seed = SEED
config.activations = [tanh, sigmoid, relu, gauss, identity, sin]
config.res_h = img_res
config.res_w = img_res
config.prob_mutate_weight = 0.5 # no weight mutation?
config.hidden_nodes_at_start = 0
config.use_input_bias = True
config.node_agg = 'sum'
config.normalize_outputs = True # True gets results faster but does worse in the long run
config.prob_add_node = .9 # very high to encourage growth
config.prob_add_connection = .8 # 
config.init_connection_probability = 1.0 # more sgd
config.num_inputs = n_features + (incl_xy*2 + config.use_radial_distance + config.use_input_bias)

# SGD
config.sgd_learning_rate = .15 # high seems to work well at least on sunrise
lr_decay = 0.9
sgd_every = 1 # anything other than 1 doesn't really make sense with this simple of an EA
config.sgd_steps = 200

#%% # Load target image
image_map = {
   'fox':       'https://live.staticflickr.com/7492/15677707699_d9d67acf9d_b.jpg',
   'sunrise':   'https://upload.wikimedia.org/wikipedia/commons/1/1e/Sunrise_over_the_sea.jpg'
}


target = imageio.imread(image_map['fox'])
# target = imageio.imread(image_map['sunrise'])
target = target[..., :3] / 255. # convert to floats and remove alpha channel

if img_res >= target.shape[0]//2:
    c = [target.shape[0]//2, target.shape[1]//2]
    r = img_res//2
    target = target[c[0]-r:c[0]+r, c[1]-r:c[1]+r] # center crop
else:
    min_dim = min(target.shape[:2])
    c = [target.shape[0]//2, target.shape[1]//2]
    r = min_dim//2
    target = target[c[0]-r:c[0]+r, c[1]-r:c[1]+r] # center crop to square
    target = cv2.resize(target, (img_res, img_res)) # resize

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
    coord_range=(-0.5, 0.5)
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
    new_pop = population[:elitism] # elitism
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
    
fits = 1.0-loss(torch.stack([ind(X) for ind in population]), target)
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
        
        # SGD
        if (gen+1) % sgd_every == 0:
            steps = sgd_weights(children, X, target, loss, config, anim_images)
            config.sgd_learning_rate *= lr_decay
            config._not_dirty() # hacky
        
        # Evaluation
        imgs = [child(X) for child in children]
        fits = 1.0-loss(torch.stack(imgs), target)
        for i, fit in enumerate(fits):
            children[i].fitness = fit
        
        # Selection
        population = tournament_selection(population + children) # sorted
        
        if make_gif:
            animate(population, children, anim_images, X, steps)
        
        pbar.set_description(f'f:{population[0].fitness.item():.4f}')
        
except KeyboardInterrupt:
    print("ending early")
except RuntimeError as e:
    print("RuntimeError:", e)
    
# do a cheeky sgd on the final champion's weights
population = sorted(population, key=lambda x: x.fitness, reverse=True)
config.sgd_steps = 100
config.sgd_learning_rate = 1e-3
sgd_weights(population[:1], X, target, loss, config, anim_images)

#%% Show result
img = population[0](X).detach().cpu()
save_image(img, f'images/final.png')
plt.imshow(img)

# %% Save gif
if make_gif:
    print("saving gif...")
    imageio.mimsave('images/evolution.gif', [img_as_ubyte(i) for i in anim_images], fps=24)
    plt.show();

