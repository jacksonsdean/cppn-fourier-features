import torch
from functorch.compile import aot_function, make_boxed_compiler
from tqdm import trange

class EarlyStopping:
    def __init__(self, patience:int=1, min_delta:float=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_loss = torch.inf

    def check_stop(self, loss:torch.Tensor) -> bool:
        if loss < self.min_loss:
            self.min_loss = loss
            self.counter = 0
        elif loss > (self.min_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def sgd_weights(genomes, inputs, target, fn, config, save_images=None, save_images_every=1, early_stop=True):
    all_params = []
    for c in genomes:
        c.prepare_optimizer()  # create parameters
        all_params.extend([cx.weight for cx in c.connection_genome.values()])

    # All CPPN weights in one optimizer
    optimizer = torch.optim.Adam(all_params, lr=config.sgd_learning_rate)

    # Compile function
    def f(X, *gs):
        return torch.stack([g(X, force_recalculate=True) for g in gs[0]])
    def fw(f,_): return f
    compiled_fn = aot_function(f, fw_compiler=make_boxed_compiler(fw))

    def save_anim(imgs, step, loss):
        if step % save_images_every != 0:
            return
        imgs_fit = zip(imgs, loss)
        imgs_fit = sorted(imgs_fit, key=lambda x: x[1], reverse=False)
        if save_images is not None:
            save_images.append(imgs_fit[0][0].detach().cpu())

    # Optimize
    pbar = trange(config.sgd_steps, desc="Compiling population AOT function... ", leave=False)
    step = 0
    stopping = EarlyStopping(patience=3 if early_stop else config.sgd_steps, min_delta=-0.01)
    for step in pbar:
        imgs = compiled_fn(inputs, genomes)
        loss = fn(imgs, target)
        save_anim(imgs, step, loss.detach())
        loss = loss.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if stopping.check_stop(loss.item()):
            break
        
        pbar.set_postfix_str(f"loss={loss.detach().clone().mean().item():.3f}")
        pbar.set_description_str("Optimizing weights")
        
    return step
        
