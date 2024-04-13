
from pathlib import Path
import torch
from cmcd.replay_buffer import ReplayBuffer
from cmcd.loss_fn import LossFunction
from cmcd.sampler import Sampler

from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    SpinnerColumn,
    TimeRemainingColumn,
)

import wandb

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

    buffer = ReplayBuffer(config["n_bridges"] + 1, config["x_dim"], 10 * config["batch_size"])

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
                repel_ratio = config["repel_percentage"]

                paths = None
                samples = None
                indicies = None

                if not config["use_buffer"]:
                    samples = sampler.sample_and_evaluate(batch_size)
                    ln_z = samples.ln_z.detach().item()
                    elbo = samples.elbo.detach().item()
                else:
                    if len(buffer) >= batch_size // 2:
                        paths = sampler.sample_and_evaluate(batch_size // 2)
                        ln_z = paths.ln_z.detach().item()
                        elbo = paths.elbo.detach().item()
                        indicies, buffer_paths = buffer.sample(batch_size // 2)
                        paths = torch.concat([buffer_paths, paths.trajectory], dim=1)
                    else:
                        paths = sampler.sample_and_evaluate(
                            batch_size, repel=repel, repel_percentage=config["repel_percentage"]
                        )
                        ln_z = paths.ln_z.detach().item()
                        elbo = paths.elbo.detach().item()
                        paths = paths.trajectory

                    paths = paths.detach()
                    samples = sampler.evaluate(paths, calc_score=True)
                    if config["use_buffer"]:
                        buffer.add(sampler, samples, indicies)
                
                loss = loss_fn.evaluate(sampler, samples)
                loss.backward()
                
                # torch.nn.utils.clip_grad_norm_(sampler.parameters(), 1)

                optim.step()
                scheduler.step()

                loss = loss.detach().item()
                all_losses.append(loss)

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
                        sampler_ln_z=sampler.ln_z.detach().item(),
                        vi=sampler.initial_dist.scale.mean().detach().item(),
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
                    sampler_ln_z=sampler.ln_z.detach().item()
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
                        vi=sampler.initial_dist.scale.mean().detach().item(),
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
        ln_z_bias=[],
        ess=[],
    )

    truth = sampler.target_density.log_norm()

    with torch.no_grad():
        for i in range(30):
            samples = sampler.sample(batch_size, repel=False, repel_percentage=0.0)
            samples = sampler.evaluate(samples)
            loss = loss_fn.evaluate(sampler, samples)
            progress_dict["eval_loss"].append(loss.detach().item())  # type: ignore
            progress_dict["ln_z"].append(samples.ln_z.detach().item())  # type: ignore
            progress_dict["ln_z_bias"].append(  # type: ignore
                (samples.ln_z.detach() - truth).abs().item()
            )
            progress_dict["elbo"].append(samples.elbo.detach().item())  # type: ignore
            progress_dict["ess"].append(samples.ess())  # type: ignore

        print("Evaluation Loss", progress_dict["eval_loss"])
        print("ln z", progress_dict["ln_z"])
        print("elbo", progress_dict["elbo"])

        wandb.summary["final_loss"] = progress_dict["eval_loss"]
        wandb.summary["final_loss_std"] = torch.tensor(progress_dict["eval_loss"]).std()
        wandb.summary["final_loss_mean"] = torch.tensor(
            progress_dict["eval_loss"]
        ).mean()

        wandb.summary["ln_z"] = progress_dict["ln_z"]
        wandb.summary["ln_z_std"] = torch.tensor(progress_dict["ln_z"]).std()
        wandb.summary["ln_z_mean"] = torch.tensor(progress_dict["ln_z"]).mean()

        wandb.summary["ln_z_bias"] = progress_dict["ln_z_bias"]
        wandb.summary["ln_z_bias_std"] = torch.tensor(progress_dict["ln_z_bias"]).std()
        wandb.summary["ln_z_bias_mean"] = torch.tensor(progress_dict["ln_z_bias"]).mean()

        wandb.summary["elbo"] = progress_dict["elbo"]
        wandb.summary["elbo_std"] = torch.tensor(progress_dict["elbo"]).std()
        wandb.summary["elbo_mean"] = torch.tensor(progress_dict["elbo"]).mean()

        wandb.summary["ess"] = progress_dict["ess"]
        wandb.summary["ess"] = torch.tensor(progress_dict["ess"]).std()
        wandb.summary["ess"] = torch.tensor(progress_dict["ess"]).mean()

        wandb.finish()

        torch.save(progress_dict, log_directory / "progress.pth")
