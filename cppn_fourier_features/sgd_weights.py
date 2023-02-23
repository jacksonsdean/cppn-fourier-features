import torch
from torchvision.transforms import Resize, Lambda
from functorch.compile import compiled_function, draw_graph, aot_function, make_boxed_compiler, make_boxed_func
from tqdm import trange


def sgd_weights(genomes, inputs, target, fn, config):
    all_params = []
    for c in genomes:
        c.prepare_optimizer()  # create parameters
        # add connection weights
        all_params.extend([cx.weight for cx in c.connection_genome.values()])

    # optimize all genome weights
    optimizer = torch.optim.Adam(all_params, lr=config.sgd_learning_rate)

    def f(*gs):
        return torch.stack([g(inputs, force_recalculate=True) for g in gs[0]])

    def fw(f,_): return f

    pbar = trange(config.sgd_steps, desc="Compiling AOT function", leave=False)

    compiled_fn = aot_function(f, fw_compiler=make_boxed_compiler(fw))
    # compiled_fn = aot_function(f, fw_compiler=fw)
    # compiled_fn = make_boxed_func(compiled_fn)
    
    pbar.set_description_str("Optimizing weights")
    
    for _ in pbar:
        imgs = compiled_fn(genomes)
        loss = fn(imgs, target).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.set_postfix_str(f"loss={loss.detach().clone().mean().item():.3f}")
        
