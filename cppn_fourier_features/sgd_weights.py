import torch
from functorch.compile import aot_function, make_boxed_compiler
from tqdm import trange
from cppn_torch.graph_util import activate_population

class EarlyStopping:
    def __init__(self, patience:int=1, min_delta:float=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_loss = torch.inf

    def check_stop(self, loss:float) -> bool:
        if loss < (self.min_loss + self.min_delta):
            self.min_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def sgd_weights(genomes, inputs, target, fn, config, save_images=None, lr_decay=1.0, save_images_every=1, early_stop=True, min_early_stop_delta=-0.0003):
    all_params = []
    for c in genomes:
        c.prepare_optimizer()  # create parameters
        all_params.extend([cx.weight for cx in c.connection_genome.values()])

    # All CPPN weights in one optimizer
    optimizer = torch.optim.Adam(all_params, lr=config.sgd_learning_rate)

    def f(X, *gs):
        if config.activation_mode == 'population':
            return activate_population(gs[0], config, X)
        else:
            return torch.stack([g(X, force_recalculate=True) for g in gs[0]])
    def fw(f,_): return f
    
    compiled_fn = f
    pbar = trange(config.sgd_steps, leave=False)
    
    if hasattr(config, 'use_aot') and config.use_aot:
        pbar.set_description_str("Compiling population AOT function... ")
        if torch.__version__.startswith("1") or config.activation_mode != 'node':
            if hasattr(config, 'use_aot') and config.use_aot:
                # super slow unless there are a ton of SGD steps
                compiled_fn = aot_function(f, fw_compiler=make_boxed_compiler(fw))
        else:
            torch._dynamo.config.verbose=True
            compiled_fn = torch.compile(f)
    

    def save_anim(imgs, step, loss):
        if step % save_images_every != 0:
            return
        imgs_fit = zip(imgs, loss)
        imgs_fit = sorted(imgs_fit, key=lambda x: x[1], reverse=False)
        if save_images is not None:
            save_images.append(imgs_fit[0][0].detach().cpu())

    # Optimize
    step = 0
    stopping = EarlyStopping(patience=3 if early_stop else config.sgd_steps, min_delta=min_early_stop_delta)
    for step in pbar:
        imgs = compiled_fn(inputs, genomes)
        loss = fn(imgs, target)
        pbar.set_postfix_str(f"pop loss: mean {loss.detach().mean().item():.3f} | min {loss.detach().min().item():.3f}")
        save_anim(imgs, step, loss.detach())
        loss = loss.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer.param_groups[0]['lr'] *= lr_decay
        
        if stopping.check_stop(loss.item()):
            break
        
        pbar.set_description_str("Optimizing weights (SGD)")
        
    return step
        
