import click
import pickle
import dnnlib
import torch
import tqdm
import numpy as np
from torch_utils import misc
from torch_utils import distributed as dist
from training import dataset
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit 


def loss_fn(net, sigma, sigma_data, images, labels):
    weight = (sigma**2 + sigma_data**2) / (sigma * sigma_data) ** 2
    n = torch.randn_like(images) * sigma
    sigma = torch.tensor(sigma, device=images.device)
    D_yn = net(images + n, sigma, labels)
    return weight * ((D_yn - images) ** 2)

def gather_dicts(dicts, rank, world_size, device='cuda'):
    """
    Gathers dictionaries with float values from all processes to rank 0.

    Args:
    - dicts (dict): Dictionary with float values to gather.
    - rank (int): The rank of the current process.
    - world_size (int): Total number of processes.
    - device (str): The device to perform the gather operation on.

    Returns:
    - (dict): The gathered dictionary on rank 0, None on other ranks.
    """
    # Convert local dict to a list of keys and a tensor of values
    local_keys = list(dicts.keys())
    local_values = torch.tensor(list(dicts.values()), dtype=torch.float32, device=device)

    # Gather the keys from all processes
    gathered_keys = [None] * world_size
    torch.distributed.all_gather_object(gathered_keys, local_keys)

    # Gather the values from all processes
    gathered_values = [torch.zeros_like(local_values) for _ in range(world_size)]
    torch.distributed.all_gather(gathered_values, local_values)

    # Only rank 0 will construct the gathered dictionary
    if rank == 0:
        # Flatten the list of keys
        all_keys = [key for sublist in gathered_keys for key in sublist]

        # Flatten the list of values
        all_values = torch.cat(gathered_values).tolist()

        # Construct the gathered dictionary
        gathered_dict = dict(zip(all_keys, all_values))
        return gathered_dict

    return None

@click.command()
@click.option(
    "--network",
    "network_pkl",
    help="Network pickle filename",
    metavar="PATH|URL",
    type=str,
    required=True,
)
@click.option(
    "--num-sigmas",
    help="Number of noise levels  [default: 100]",
    metavar="INT",
    type=click.IntRange(min=1),
    default=100,
)
@click.option(
    "--sigma_min",
    help="Lowest noise level  [default: varies]",
    metavar="FLOAT",
    type=click.FloatRange(min=0, min_open=True),
    default=0.002,
)
@click.option(
    "--sigma_max",
    help="Highest noise level  [default: varies]",
    metavar="FLOAT",
    type=click.FloatRange(min=0, min_open=True),
    default=80.0,
)
@click.option(
    "--sigma_data",
    help="Data noise level  [default: 0.5]",
    metavar="FLOAT",
    type=click.FloatRange(min=0, min_open=True),
    default=0.5,
)
@click.option(
    "--data", help="Path to the dataset", metavar="ZIP|DIR", type=str, required=True
)
@click.option(
    "--cond",
    help="Train class-conditional model",
    metavar="BOOL",
    type=bool,
    default=False,
    show_default=True,
)
@click.option(
    "--xflip",
    help="Enable dataset x-flips",
    metavar="BOOL",
    type=bool,
    default=False,
    show_default=True,
)
@click.option(
    "--cache",
    help="Cache dataset in CPU memory",
    metavar="BOOL",
    type=bool,
    default=True,
    show_default=True,
)
@click.option(
    "--batch",
    help="Total batch size",
    metavar="INT",
    type=click.IntRange(min=1),
    default=512,
    show_default=True,
)
@click.option(
    "--batch-per-sigma",
    help="Number of batches per sigma  [default: 1]",
    metavar="INT",
    type=click.IntRange(min=1),
    default=10,
)
@click.option("--seed", help="Random seed  [default: random]", metavar="INT", type=int)
@click.option("--output-path", help="Output path", type=str, required=True)
def main(
    network_pkl,
    data,
    cond,
    xflip,
    cache,
    batch,
    batch_per_sigma,
    seed,
    num_sigmas,
    sigma_min,
    sigma_max,
    sigma_data,
    output_path,
    device=torch.device("cuda"),
):
    dist.init()
    if seed is None:
        seed = np.random.randint(1 << 31)
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    dist.print0(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
        net = pickle.load(f)[f"ema_0.034"].to(device)
    batch_gpu = batch // dist.get_world_size()
    dataset_obj = dataset.ImageFolderDataset(
        path=data,
        use_labels=cond,
        xflip=xflip,
        cache=cache,
    )
    dataset_sampler = misc.InfiniteSampler(
        dataset=dataset_obj,
        rank=dist.get_rank(),
        num_replicas=dist.get_world_size(),
        seed=seed,
    )
    dataset_iterator = iter(
        torch.utils.data.DataLoader(
            dataset=dataset_obj,
            sampler=dataset_sampler,
            batch_size=batch_gpu,
            pin_memory=True,
            num_workers=1,
            prefetch_factor=2,
        )
    )
    sigmas = np.exp(np.linspace(np.log(sigma_min), np.log(sigma_max), num_sigmas))
    rank_sigmas = sigmas[dist.get_rank() :: dist.get_world_size()]
    sigma_loss = {}
    for sigma in tqdm.tqdm(rank_sigmas, desc="Sigma", disable=(dist.get_rank() != 0)):
        loss = []
        for _ in tqdm.tqdm(
            range(batch_per_sigma),
            desc="Batch",
            leave=False,
            disable=(dist.get_rank() != 0),
        ):
            images, labels = next(dataset_iterator)
            images = images.to(device).to(torch.float32) / 127.5 - 1
            images = (
                (images - net.data_mean[None, :, None, None])
                / net.data_std[None, :, None, None]
                * net.sigma_data
            )
            labels = labels.to(device)
            loss.append(loss_fn(net, sigma, sigma_data, images, labels))
        sigma_loss[sigma] = torch.cat(loss).mean().item()
    # gather sigma_loss from all ranks to rank 0
    torch.distributed.barrier()
    all_sigma_losses = gather_dicts(sigma_loss, dist.get_rank(), dist.get_world_size(), device)
    if dist.get_rank() == 0:
        matplotlib.use("Agg")
        # sort the dict
        all_sigma_losses = dict(sorted(all_sigma_losses.items()))
        for sigma, loss in all_sigma_losses.items():
            print(f"sigma: {sigma:.2f} : {loss:.6f}")
        # mark the minimum loss
        min_loss = min(all_sigma_losses.values())
        min_sigma = min(all_sigma_losses, key=all_sigma_losses.get)
        plt.scatter(min_sigma, min_loss, color="red")
        # mark its sigma and loss value
        plt.text(min_sigma, min_loss, f"({min_sigma:.2f}, {min_loss:.6f})", fontsize=12)
        # fit the reciprocal of loss with a gaussian distribution
        losses = np.array(list(all_sigma_losses.values()))
        loss_reciprocal = 1 / losses
        sigma_log = np.array([np.log(sigma) for sigma in all_sigma_losses.keys()])
        # fit the mu and sigma for the gaussian distribution, 
        def gaussian(x, a, mu, sigma):
            return a * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
        popt, _ = curve_fit(gaussian, sigma_log, loss_reciprocal)
        a, mu, sigma = popt
        # plot the fitted gaussian distribution
        plt.plot(*zip(*sorted(all_sigma_losses.items())), label="loss")
        plt.xlabel("sigma")
        plt.ylabel("loss")
        plt.xscale("log")
        y = gaussian(sigma_log, a, mu, sigma)
        plt.plot(np.exp(sigma_log), 1 / losses, label="1 / loss")
        plt.plot(np.exp(sigma_log), y, label=f"a: {a:.2f}, mu: {mu:.2f}, sigma: {sigma:.2f}")
        # plot y * loss
        plt.plot(np.exp(sigma_log), y * losses, label="y * loss")
        plt.legend()
        plt.savefig(output_path)


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------
