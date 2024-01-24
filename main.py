import datetime
import logging
from cmcd.initial_dist import prepare_init_dist
from cmcd.loss_fn import prepare_loss_fn
from cmcd.sampler import CMCD
from cmcd.score import prepare_score_fn
from cmcd.densities import prepare_target
from cmcd.anneal import prepare_anneal

import torch
import click
from pathlib import Path
import random
import numpy as np
import yaml
import wandb

from train import train

# torch._dynamo.config.cache_size_limit = 256

def setup(config: dict):
    score_fn = prepare_score_fn(config)
    init_dist = prepare_init_dist(config)
    loss_fn = prepare_loss_fn(config)
    target = prepare_target(config)
    anneal = prepare_anneal(config)
    sampler = CMCD(
        init_dist,
        target,
        anneal,
        score_fn,
        config["n_bridges"],
        config["init_eps"],
        config["eps_trainable"],
    )

    params = list(filter(lambda kv: kv[0] != "ln_z", sampler.named_parameters()))
    params = [x[1] for x in params]

    optim = torch.optim.Adam(
        [{"params": params, "lr": config['lr']}, {"params": sampler.ln_z, "lr": 0.1}],
        lr=config["lr"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=config['max_epoch'])

    return sampler, optim, scheduler, loss_fn


@click.command()
@click.option("--x_dim", type=int)
@click.option("--t_dim", type=int)
@click.option("--h_dim", type=int)
@click.option("--n_bridges", type=int)
@click.option("--target", type=str)
@click.option("--anneal", type=str)
@click.option("--max_epoch", type=int)
@click.option("--batch_size", type=int)
@click.option("--init_eps", type=float)
@click.option("--eps_trainable", type=bool)
@click.option("--lr", type=float)
@click.option("--seed", type=int)
@click.option("--correct", type=bool)
@click.option("--repel", type=bool)
@click.option("--loss_fn", type=str)
@click.option("--path", type=str)
@click.option("--dw_d", type=int)
@click.option("--dw_m", type=int)
@click.option("--dw_delta", type=int)
@click.option("--repel_percentage", type=float)
@click.option("--progress", type=int)
@click.option("--pretrain-epoch", type=int)
@click.option("--use_buffer", type=bool)
def main(
    x_dim: int,
    t_dim: int,
    h_dim: int,
    n_bridges: int,
    target: str,
    anneal: str,
    max_epoch: int,
    batch_size: int,
    init_eps: float,
    eps_trainable: bool,
    lr: float,
    seed: int,
    correct: bool,
    repel: bool,
    loss_fn: str,
    path: str,
    dw_d: int,
    dw_m: int,
    dw_delta: int,
    repel_percentage: float,
    progress: int,
    pretrain_epoch: int,
    use_buffer: bool,
):
    torch.set_default_device("cuda")

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    now = datetime.datetime.now()
    current_time = now.strftime("%y-%m-%d-%H:%M:%S")
    log_directory = Path(".", "logs", target, current_time)
    log_directory.mkdir()

    config = dict(
        x_dim=x_dim,
        h_dim=h_dim,
        t_dim=t_dim,
        n_bridges=n_bridges,
        target=target,
        anneal=anneal,
        max_epoch=max_epoch,
        batch_size=batch_size,
        init_eps=init_eps,
        eps_trainable=eps_trainable,
        lr=lr,
        correct=correct,
        repel=repel,
        loss_fn=loss_fn,
        dw_d=dw_d,
        dw_m=dw_m,
        dw_delta=dw_delta,
        repel_percentage=repel_percentage,
        pretrain_epoch=pretrain_epoch,
        use_buffer=use_buffer
    )

    wandb.init(project="dissertation", config=config, entity="brianlee-lck")
    wandb.run.log_code(".")  # type: ignore

    # store h_params
    with open(log_directory / "hparam.yaml", "w") as file:
        yaml.dump(config, file)

    sampler, optim, scheduler, loss_fn_cls = setup(config)
    wandb.watch(sampler)

    if path != None:
        progress_path = f"{path}/progress"
        if progress != None:
            progress_path = f"{progress_path}_{progress}"
        progress_path = f"{progress_path}.pth"

        checkpoint = torch.load(progress_path)
        sampler.load_state_dict(checkpoint["sampler"])
        optim.load_state_dict(checkpoint["optim"])
        if scheduler in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        else:
            logging.warning("checkpoint does not have scheduler")

    train(
        max_epoch,
        config,
        batch_size,
        sampler,
        optim,
        scheduler,
        loss_fn_cls,
        log_directory,
    )


if __name__ == "__main__":
    main()
