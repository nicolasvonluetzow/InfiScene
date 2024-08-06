import math
import os
import random
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..diffusion.scene_dataset import SceneDataset, chunk_collate_fn
from ..diffusion.utils import get_all_files, mc_and_save, set_random_seeds
from .model_autoencoder import ConvAutoencoder


def train_autoencoder(args):
    set_random_seeds()

    def print_and_log(*args):
        print(*args)
        print(*args, file=log_file)

    run_path = os.path.join(args.out, args.name)
    checkpoint_path = os.path.join(run_path, 'model.chkpt')

    subrun_path = os.path.join(run_path, 'sub0')
    i = 0
    while os.path.exists(subrun_path):
        i += 1
        subrun_path = os.path.join(run_path, f'sub{i}')

    os.makedirs(subrun_path, exist_ok=True)

    image_path = os.path.join(subrun_path, 'images')
    os.makedirs(image_path, exist_ok=True)

    log_file = open(os.path.join(subrun_path, 'log.txt'), 'w')

    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(
        args.log_dir, args.name, f"sub{i}"))

    print_and_log("Subrun:", i)
    print_and_log("Args:", args)
    print_and_log("\n---\n")
    del i

    files = get_all_files(args.data)
    train_files = files[:int(0.9 * len(files))][:args.max_files]
    val_files = files[int(0.9 * len(files)):][:args.max_files]

    rep = args.min_epoch_samples // len(train_files) if len(
        train_files) < args.min_epoch_samples else 1
    train_files = train_files * rep

    train_dataset = SceneDataset(
        train_files, True, args.n_chunks, args.chunk_size, False)
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=chunk_collate_fn, pin_memory=True, num_workers=4)

    val_dataset = SceneDataset(
        val_files, False, args.n_chunks, args.chunk_size, False)
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size, collate_fn=chunk_collate_fn, pin_memory=True, num_workers=4)

    print_and_log(
        f"Total files: \t {len(train_files) + len(val_files)} \nTrain/Val: \t {len(train_files)}/{len(val_files)}")
    assert len(train_files) > 0 and len(val_files) > 0

    if rep != 1:
        print_and_log(
            f"Epoch size:  \t {len(train_files):,} ({rep} repetitions per epoch).")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = ConvAutoencoder(latent_channels=args.latent_channels,
                            process_channels=args.process_channels,
                            num_scaling_blocks=args.n_blocks).to(device)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))

    def count_params(module):
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    print_and_log(f"Model parameters: {count_params(model):,}")
    del count_params

    print_and_log("\n---\n")

    reconstruction_loss = torch.nn.MSELoss()

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda epoch: max(math.pow(0.5, int(epoch / args.lr_schedule_interval)), 1e-8), verbose=True)

    step = 0
    val_step = 0

    accum_iter = args.accum_iter

    for epoch in range(args.epochs):
        model.train()
        epoch_losses = []
        for batch in tqdm(train_dataloader):
            data = batch["chunks"].to(device)

            reconstruction, mean, logvar = model(data)
            writer.add_scalar("latent/mean", mean.mean().item(), step)
            writer.add_scalar(
                "latent/std", torch.exp(0.5 * logvar).mean().item(), step)

            loss = reconstruction_loss(reconstruction, data)
            writer.add_scalar("Loss/reconstruction", loss.item(), step)

            kl_loss = -0.5 * \
                torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
            writer.add_scalar("Loss/kl", kl_loss.item(), step)

            loss = loss + args.var_weight * kl_loss

            writer.add_scalar("Loss/train", loss.item(), step)
            epoch_losses.append(loss.item())

            loss.backward()

            if (step + 1) % accum_iter == 0 or step == len(train_dataloader) - 1:
                opt.step()
                opt.zero_grad()

            step += 1

        mean_train_loss = sum(epoch_losses) / len(epoch_losses)
        writer.add_scalar("Loss/train_epoch", mean_train_loss, epoch)

        print_and_log(
            f"Finished epoch {epoch + 1}/{args.epochs}")
        print_and_log(f"Train loss: \t {mean_train_loss:05f}")

        scheduler.step()

        with torch.no_grad():
            model.eval()

            val_epoch_losses = []
            for batch in val_dataloader:
                data = batch["chunks"].to(device)

                reconstruction, mean, logvar = model(data)

                loss = reconstruction_loss(reconstruction, data)
                writer.add_scalar("val/reconstruction", loss.item(), val_step)

                kl_loss = -0.5 * \
                    torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
                writer.add_scalar("val/kl", kl_loss.item(), val_step)

                loss = loss + args.var_weight * kl_loss

                writer.add_scalar("val/train", loss.item(), val_step)
                val_epoch_losses.append(loss.item())

                val_step += 1

            mean_val_loss = sum(val_epoch_losses) / len(val_epoch_losses)
            writer.add_scalar("val/val_epoch", mean_val_loss, epoch)
            print_and_log(f"Val loss: \t {mean_val_loss:05f}")

            if args.sample_freq != 0 and epoch % args.sample_freq == args.sample_freq - 1:
                chosen_chunks = random.choice(val_dataset)
                chunk = chosen_chunks["chunks"][:4].to(device)

                reconstruction, _, _ = model(chunk.clone())

                for i, (c, r) in enumerate(zip(chunk, reconstruction)):
                    p = os.path.join(
                        image_path, f"{epoch:04d}-{i:03d}_gt.obj")
                    mc_and_save(c, p)
                    p = os.path.join(
                        image_path, f"{epoch:04d}-{i:03d}_recon.obj")
                    mc_and_save(r, p)

        model.train()
        torch.save(model.state_dict(), os.path.join(run_path, "model.chkpt"))

    log_file.close()
    writer.close()
    print("Done.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--name", type=str, required=True)

    parser.add_argument("--data", type=str, required=True,
                        help='Path to directory containing training SDFs.')
    parser.add_argument("--out", type=str,
                        default=os.path.join("..", "output"), help='Path to output directory.')
    parser.add_argument("--log_dir", type=str,
                        default=os.path.join("..", "logs"), help='Path to log directory.')

    parser.add_argument("--batch_size", type=int,
                        default=1, help='Batch size.')
    parser.add_argument("--accum_iter", type=int,
                        default=1, help='Number of gradient accumulation steps.')

    parser.add_argument("--max_files", type=int, default=None,
                        help='Maximum number of files to use for training and validation.')
    parser.add_argument("--min_epoch_samples", type=int,
                        default=0, help='Minimum number of samples per epoch.')
    parser.add_argument("--sample_freq", type=int, default=1,
                        help="Interval to generate sample meshes for validation. Can be zero for no sampling.")

    parser.add_argument("--epochs", type=int, default=1000,
                        help='Number of epochs.')
    parser.add_argument("--lr", type=float, default=1e-4,
                        help='Learning rate.')
    parser.add_argument("--lr_schedule_interval", type=int, default=5,
                        help='Number of epochs between learning rate decay.')
    parser.add_argument("--var_weight", type=float,
                        default=0.005, help='Weight of KL loss.')

    parser.add_argument("--n_chunks", type=int, default=1,
                        help='Number of chunks per sample.')
    parser.add_argument("--chunk_size", type=int,
                        default=64, help='Possible chunk sizes.')

    parser.add_argument("--latent_channels", type=int, default=1)
    parser.add_argument("--process_channels", type=int, default=64)
    parser.add_argument("--n_blocks", type=int, default=1)

    args = parser.parse_args()
    train_autoencoder(args)
