import datetime
import logging
from mcmd.initial_dist import prepare_init_dist
from mcmd.loss_fn import LossFunction, ReverseKL, prepare_loss_fn
from mcmd.sampler import CMCD, Sampler
from mcmd.score import prepare_score_fn
from mcmd.densities import (
    prepare_target,
)
from mcmd.anneal import GeometricAnnealing, prepare_anneal

import torch
import click
from pathlib import Path
import random
import numpy as np

from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    SpinnerColumn,
    TimeRemainingColumn,
)
import yaml
import wandb

torch.set_default_device("cuda")


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

    optim = torch.optim.Adam(sampler.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=1000)

    return sampler, optim, scheduler, loss_fn


def grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None and torch.is_tensor(p.grad.data):
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1.0 / 2)
    return total_norm



def train(
    max_epoch: int,
    config: dict,
    batch_size: int,
    sampler: Sampler,
    optim: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    loss_fn: LossFunction,
    log_directory: Path,
):
    all_losses = []

    try:
        with Progress(
            TextColumn("{task.completed} of {task.total}"),
            BarColumn(),
            SpinnerColumn(),
            TextColumn("Loss: {task.fields[loss]};"),
            TextColumn("elbo: {task.fields[elbo]};"),
            TextColumn("ln z: {task.fields[ln_z]};"),
            TextColumn("eps: {task.fields[eps]};"),
            TextColumn("vi: {task.fields[vi]};"),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task(
                "Training...",
                total=max_epoch,
                elbo=None,
                ln_z=None,
                loss=None,
                eps=None,
                grid_t_min=None,
                vi=None,
            )

            for epoch in range(max_epoch):
                optim.zero_grad()

                repel = config["repel"] and (
                    config["pretrain_epoch"] == None or epoch < config["pretrain_epoch"]
                )

                samples = sampler.sample(batch_size, repel=repel, repel_percentage=config['repel_percentage'])

                loss = loss_fn.evaluate(samples)
                loss.backward()

                optim.step()
                scheduler.step()

                loss = loss.detach().item()
                all_losses.append(loss)

                ln_z = samples.ln_z.detach().item()
                elbo = samples.elbo.detach().item()
                eps = sampler.eps.item()

                wandb.log(
                    dict(
                        loss=loss,
                        ln_z=ln_z,
                        elbo=elbo,
                        eps=eps,
                        beta_min=sampler.betas().min().detach().item(),
                        beta_max=sampler.betas().max().detach().item(),
                        betas=sampler.betas().tolist(),
                        grad_norm=grad_norm(sampler),
                        grid_t_min=sampler.grid_t.min().detach().item(),
                        grid_t_max=sampler.grid_t.max().detach().item(),
                    )
                )

                progress.update(
                    task,
                    advance=1,
                    loss=loss,
                    ln_z=ln_z,
                    elbo=elbo,
                    vi=sampler.initial_dist.scale.mean().detach().item(),
                    eps=eps,
                )

                if epoch % 1000 == 0:
                    progress_dict = dict(
                        epoch=epoch,
                        samples=samples,
                        all_losses=all_losses,
                        sampler=sampler.state_dict(),
                        optim=optim.state_dict(),
                        scheduler=scheduler.state_dict(),
                        eval_loss=[],
                        elbo=[],
                        ln_z=[],
                    )
                    torch.save(progress_dict, log_directory / f"progress_{epoch}.pth")

    except KeyboardInterrupt:
        pass

    progress_dict = dict(
        all_losses=all_losses,
        sampler=sampler.state_dict(),
        optim=optim.state_dict(),
        scheduler=scheduler.state_dict(),
        eval_loss=[],
        elbo=[],
        ln_z=[],
    )

    truth = sampler.target.log_norm()
    kl = ReverseKL()
    with torch.no_grad():
        for i in range(30):
            samples = sampler.sample(2000, repel=False, repel_percentage=0.0)
            loss = loss_fn.evaluate(samples)
            progress_dict["eval_loss"].append(loss.detach().item()) # type: ignore
            progress_dict["ln_z"].append(samples.ln_z.detach().item()) # type: ignore
            progress_dict["ln_z_bias"].append( # type: ignore
                (samples.ln_z.detach() - truth).abs().item()
            )
            progress_dict["elbo"].append(samples.elbo.detach().item()) # type: ignore
            progress_dict["ess"].append(samples.ess()) # type: ignore

        print("Evaluation Loss", progress_dict["eval_loss"])
        print("ln z", progress_dict["ln_z"])
        print("elbo", progress_dict["elbo"])

        wandb.summary["final_loss"] = progress_dict["eval_loss"]
        wandb.summary["final_loss_std"] = torch.tensor(progress_dict["eval_loss"]).std()
        wandb.summary["final_loss_mean"] = torch.tensor(progress_dict["eval_loss"]).mean()

        wandb.summary["ln_z"] = progress_dict["ln_z"]
        wandb.summary["ln_z_std"] = torch.tensor(progress_dict["ln_z"]).std()
        wandb.summary["ln_z_mean"] = torch.tensor(progress_dict["ln_z"]).mean()

        wandb.summary["elbo"] = progress_dict["elbo"]
        wandb.summary["elbo_std"] = torch.tensor(progress_dict["elbo"]).std()
        wandb.summary["elbo_mean"] = torch.tensor(progress_dict["elbo"]).mean()

        wandb.summary["ess"] = progress_dict["ess"]
        wandb.summary["ess"] = torch.tensor(progress_dict["ess"]).std()
        wandb.summary["ess"] = torch.tensor(progress_dict["ess"]).mean()

        wandb.finish()

        torch.save(progress_dict, log_directory / "progress.pth")


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
@click.option("--a", type=float)
@click.option("--progress", type=int)
@click.option("--pretrain-epoch", type=int)
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
    a: float,
    progress: int,
    pretrain_epoch: int,
):
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
        a=a,
        pretrain_epoch=pretrain_epoch,
    )

    wandb.init(project="dissertation", config=config, entity="brianlee-lck")
    wandb.run.log_code(".") # type: ignore
    
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

    train(max_epoch, config, batch_size, sampler, optim, scheduler, loss_fn_cls, log_directory)


if __name__ == "__main__":
    main()
