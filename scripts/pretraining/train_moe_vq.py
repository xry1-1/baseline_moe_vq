"""Single-device MOMENT + MoE + input VQ pretraining, with no PILE/W&B dependency.

Data utilities deliberately import only NumPy; model dependencies load in main().
Run from the repository root: python scripts/pretraining/train_moe_vq.py --help
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def load_array(path, layout="TC"):
    """Read explicit [time, channels] CSV/NPY or [windows, channels, length] NPY."""
    path = Path(path)
    if path.suffix.lower() == ".npy":
        return np.load(path, allow_pickle=False)
    if path.suffix.lower() != ".csv":
        raise ValueError("Data must be a header-bearing .csv or a numeric .npy file.")
    if layout != "TC":
        raise ValueError("CSV requires layout TC ([time, channels]).")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        names = reader.fieldnames
        rows = list(reader)
    if not names or not rows:
        raise ValueError("CSV must contain a header and at least one data row.")
    if len(set(names)) != len(names):
        raise ValueError("CSV column names must be unique.")
    columns = []
    for name in names:
        values = [row.get(name, "") for row in rows]
        parsed, valid = [], []
        for value in values:
            try:
                parsed.append(float(value))
                valid.append(True)
            except (TypeError, ValueError):
                parsed.append(0.)
                valid.append(False)
        if all(valid):
            columns.append(parsed)
        elif any(valid):
            raise ValueError(f"CSV numeric column {name!r} contains missing/non-numeric values.")
        # Entirely nonnumeric columns (for example ISO date strings) are excluded.
    if not columns:
        raise ValueError("CSV contains no complete numeric channels.")
    return np.asarray(columns, dtype=np.float32).T


def synthetic_series(length=8192, channels=2, seed=13):
    if length <= 0 or channels <= 0:
        raise ValueError("Synthetic length and channels must be positive.")
    rng = np.random.default_rng(seed)
    t = np.arange(length, dtype=np.float32)
    values = [np.sin(t/(9.+3*c)) + .3*np.cos(t/(31.+c)) +
              .0005*t + rng.normal(0., .03, length) for c in range(channels)]
    return np.asarray(values, dtype=np.float32).T


def prepare_data(raw, *, layout="TC", seq_len=512, stride=512,
                 train_fraction=.8, val_fraction=.1, standardize=True):
    """Split chronology first, fit channel scaling on train, then form windows.

    For already-windowed NCL input the first axis must be chronological and the
    caller must supply windows that do not overlap across split boundaries.
    """
    if layout not in ("TC", "NCL"):
        raise ValueError("layout must be explicit TC or NCL.")
    expected_ndim = 2 if layout == "TC" else 3
    raw = np.asarray(raw)
    if raw.ndim != expected_ndim:
        raise ValueError(f"layout {layout} requires {expected_ndim} dimensions; got {raw.shape}.")
    if not np.issubdtype(raw.dtype, np.number) or np.iscomplexobj(raw):
        raise ValueError("Data must contain real numeric values.")
    if any(dim == 0 for dim in raw.shape) or not np.isfinite(raw).all():
        raise ValueError("Data must be nonempty and contain only finite values.")
    if seq_len <= 0 or stride <= 0:
        raise ValueError("seq_len and stride must be positive.")
    if not (0 < train_fraction < 1 and 0 < val_fraction < 1 and
            train_fraction + val_fraction < 1):
        raise ValueError("Train, validation and test fractions must all be positive and sum to 1.")
    if layout == "NCL" and raw.shape[-1] != seq_len:
        raise ValueError(f"NCL window length {raw.shape[-1]} must equal seq_len={seq_len}.")
    raw = raw.astype(np.float32)
    if not np.isfinite(raw).all():
        raise ValueError("Data must remain finite when converted to float32.")
    length = len(raw)
    bounds = [0, int(length*train_fraction), int(length*(train_fraction+val_fraction)), length]
    pieces = [raw[bounds[i]:bounds[i+1]] for i in range(3)]
    for name, piece in zip(("train", "validation", "test"), pieces):
        minimum = seq_len if layout == "TC" else 1
        if len(piece) < minimum:
            raise ValueError(f"{name} split has {len(piece)} rows/windows; need at least "
                             f"{minimum} (seq_len={seq_len}). Supply more data or change splits.")
    axes = 0 if layout == "TC" else (0, 2)
    # Float64 fitting avoids cancellation/overflow; stored transforms are float32.
    mean = pieces[0].mean(axis=axes, dtype=np.float64).astype(np.float32)
    std = pieces[0].std(axis=axes, dtype=np.float64).astype(np.float32)
    std = np.where(std < 1e-6, 1., std).astype(np.float32)
    if not standardize:
        mean, std = np.zeros_like(mean), np.ones_like(std)
    result = {"mean": mean, "std": std, "split_bounds": bounds, "layout": layout}
    for name, piece in zip(("train", "validation", "test"), pieces):
        if layout == "TC":
            transformed = (piece-mean)/std
            windows = [transformed[i:i+seq_len].T
                       for i in range(0, len(piece)-seq_len+1, stride)]
            result[name] = np.ascontiguousarray(windows, dtype=np.float32)
        else:
            result[name] = np.ascontiguousarray((piece-mean[None, :, None])/std[None, :, None])
        if not np.isfinite(result[name]).all():
            raise ValueError(f"{name} transformation produced nonfinite values.")
    return result


def epoch_order(size, seed, epoch):
    """A local NumPy RNG makes shuffle order independent of model/validation RNG."""
    return np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(size)


def validate_config(config):
    model, data, train = config["model"], config.get("data", {}), config["training"]
    seq, patch = int(model["seq_len"]), int(model["patch_len"])
    if patch <= 0 or seq % patch or seq < 2*patch:
        raise ValueError("seq_len must be a multiple of patch_len with at least two patches.")
    if int(model.get("patch_stride_len", patch)) != patch:
        raise ValueError("This reconstruction trainer requires patch_stride_len == patch_len.")
    ratio = float(model.get("mask_ratio", .3))
    if not 0 < ratio < 1 or not 0 < math.ceil((seq//patch)*(1-ratio)) < seq//patch:
        raise ValueError("mask_ratio must leave at least one visible and one masked patch.")
    if int(train.get("batch_size", 16)) <= 0 or int(train.get("max_steps", 10000)) <= 0:
        raise ValueError("batch_size and max_steps must be positive.")
    if int(data.get("stride", seq)) <= 0:
        raise ValueError("Window stride must be positive.")
    if int(train.get("checkpoint_every", 100)) <= 0:
        raise ValueError("checkpoint_every must be positive.")
    if int(train.get("max_consecutive_overflows", 100)) <= 0:
        raise ValueError("max_consecutive_overflows must be positive.")


def smoke_config(config):
    """Small offline graph; independent of external model files or datasets."""
    config = copy.deepcopy(config)
    config["model"].update(seq_len=32, patch_len=8, patch_stride_len=8, mask_ratio=.5,
        d_model=32, transformer_backbone="t5-small", transformer_type="encoder_only",
        randomly_initialize_backbone=True, dropout=0., head_dropout=0.,
        vq_latent_dim=8, vq_codebook_size=16,
        transformer_config={"d_model": 32, "d_ff": 64, "num_layers": 1,
                            "num_heads": 2, "d_kv": 16, "dropout_rate": 0.,
                            "feed_forward_proj": "relu", "vocab_size": 32})
    config.setdefault("data", {}).update(stride=16, synthetic_length=512,
                                         synthetic_channels=2, layout="TC")
    config["training"].update(batch_size=2, max_steps=2, checkpoint_every=1,
        validation_every=1, max_validation_batches=2, num_threads=1, amp=False)
    return config


def capture_rng(torch):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state, torch):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def array_digest(raw):
    array = np.ascontiguousarray(raw)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode())
    digest.update(str(array.dtype).encode())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def log_event(handle, event, **fields):
    record = {"event": event, **fields}
    line = json.dumps(record, ensure_ascii=False, allow_nan=False)
    print(line, flush=True)
    handle.write(line+"\n")
    handle.flush()


def optimizer_update(loss, parameters, optimizer, scaler, max_grad_norm=1.):
    """Backpropagate and report whether GradScaler performed an optimizer update."""
    import torch

    parameters = list(parameters)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not gradients:
        raise RuntimeError("Training loss produced no parameter gradients.")
    finite_gradients = bool(torch.stack([torch.isfinite(gradient).all()
                                        for gradient in gradients]).all())
    if finite_gradients:
        # A nonfinite norm from otherwise finite gradients is still a hard error.
        grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm,
                                                        error_if_nonfinite=True))
    elif scaler.is_enabled():
        # unscale_ has already recorded the overflow for scaler.step(). Do not
        # clip Inf gradients: multiplying Inf by a zero clip coefficient creates
        # NaNs, and strict clipping would raise before the scale can back off.
        grad_norm = None
    else:
        raise FloatingPointError("Nonfinite gradients without an enabled GradScaler.")
    previous_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    updated = not scaler.is_enabled() or scaler.get_scale() >= previous_scale
    return {"optimizer_updated": updated, "grad_norm": grad_norm,
            "amp_overflow": not updated,
            "amp_scale_before": previous_scale, "amp_scale": scaler.get_scale()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/pretraining/moe_vq.yaml")
    parser.add_argument("--data", type=Path, help="CSV [T,C] or NPY; layout is set in YAML")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"checkpoints/moe_vq")
    parser.add_argument("--max-steps", type=int, help="Total optimizer-step target, including resumed steps")
    parser.add_argument("--resume", type=Path, help="Trusted checkpoint; restores full optimizer/RNG/cursor")
    parser.add_argument("--device", help="cpu, cuda or cuda:0 (one GPU only)")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Tiny offline config; implies synthetic unless --data")
    parser.add_argument("--build-memory-only", action="store_true", help="Requires --resume; fresh full train-only pass")
    args = parser.parse_args(argv)
    if args.build_memory_only and args.resume is None:
        parser.error("--build-memory-only requires --resume")
    if args.data and args.synthetic:
        parser.error("Choose --data or --synthetic")

    import yaml
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    # torch.load(..., weights_only=False) is necessary for optimizer and RNG objects.
    # Only load checkpoints created by yourself/a trusted source.
    resumed = torch.load(args.resume, map_location="cpu", weights_only=False) if args.resume else None
    if resumed:
        config = copy.deepcopy(resumed["config"])
    else:
        with args.config.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        if args.smoke:
            config = smoke_config(config)
    if args.max_steps is not None:
        config["training"]["max_steps"] = args.max_steps
    validate_config(config)
    model_config = config["model"]
    if args.build_memory_only and not model_config.get("use_vq", False):
        parser.error("--build-memory-only requires model.use_vq=True in the checkpoint")
    data_config = config.setdefault("data", {})
    train_config = config["training"]
    seed = int(train_config.get("seed", 13))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(int(train_config.get("num_threads", 4)))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested, but this PyTorch installation has no available CUDA device.")
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Only a single CPU or CUDA device is supported.")
    model_config["device"] = str(device)

    if args.data:
        source = {"kind": "file", "path": str(args.data.resolve())}
    elif args.synthetic or args.smoke:
        source = {"kind": "synthetic", "length": int(data_config.get("synthetic_length", 8192)),
                  "channels": int(data_config.get("synthetic_channels", 2)), "seed": seed}
    elif resumed:
        source = resumed["data_source"]
    elif data_config.get("path"):
        source = {"kind": "file", "path": str(Path(data_config["path"]).resolve())}
    else:
        parser.error("Supply --data PATH or --synthetic (or --smoke)")
    if source["kind"] == "synthetic":
        raw = synthetic_series(source["length"], source["channels"], source["seed"])
    else:
        raw = load_array(source["path"], data_config.get("layout", "TC"))
    fingerprint = array_digest(raw)
    if resumed and fingerprint != resumed["data_fingerprint"]:
        raise ValueError("Resume data content/shape changed; exact cursor replay requires identical data.")
    data = prepare_data(raw, layout=data_config.get("layout", "TC"),
        seq_len=int(model_config["seq_len"]), stride=int(data_config.get("stride", model_config["seq_len"])),
        train_fraction=float(data_config.get("train_fraction", .8)),
        val_fraction=float(data_config.get("val_fraction", .1)),
        standardize=bool(data_config.get("standardize", True)))
    if resumed:
        for name in ("mean", "std"):
            if not np.array_equal(data[name], resumed["transforms"][name]):
                raise ValueError("Resume train-only transforms differ from saved checkpoint.")

    sys.path.insert(0, str(ROOT))
    from moment.models.moment import MOMENT
    model = MOMENT(configs=model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_config.get("learning_rate", 1e-4)),
                                 weight_decay=float(train_config.get("weight_decay", .01)))
    use_amp = bool(train_config.get("amp", True)) and device.type == "cuda"
    amp_name = str(train_config.get("amp_dtype", "float16"))
    if amp_name not in ("float16", "bfloat16"):
        raise ValueError("amp_dtype must be float16 or bfloat16")
    amp_dtype = getattr(torch, amp_name)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    step, epoch, batch_cursor, consecutive_overflows = 0, 0, 0, 0
    if resumed:
        model.load_state_dict(resumed["model"])
        optimizer.load_state_dict(resumed["optimizer"])
        # Optimizer moments follow parameters to the selected device.
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
        if resumed.get("scaler") and scaler.is_enabled():
            scaler.load_state_dict(resumed["scaler"])
        step, epoch, batch_cursor = resumed["step"], resumed["epoch"], resumed["batch_cursor"]
        consecutive_overflows = int(resumed.get("consecutive_overflows", 0))
        restore_rng(resumed["rng"], torch)

    batch_size = int(train_config.get("batch_size", 16))
    # Local generators prevent DataLoader base_seed creation from consuming model RNG.
    def loader(values, indices=None):
        if indices is not None:
            values = values[indices]
        generator = torch.Generator().manual_seed(seed)
        return DataLoader(TensorDataset(torch.from_numpy(values)), batch_size=batch_size,
                          shuffle=False, drop_last=False, num_workers=0, generator=generator)

    def autocast():
        return torch.autocast("cuda", dtype=amp_dtype) if use_amp else nullcontext()

    def losses(outputs, x):
        mask = (1-outputs.pretrain_mask).to(x.dtype).unsqueeze(1).expand_as(x)
        count = mask.sum()
        if count.item() <= 0:
            raise ValueError("Batch has no hidden target samples; adjust patch/mask configuration.")
        mse = ((outputs.reconstruction.float()-x.float()).square()*mask).sum()/count
        aux = outputs.aux_loss
        if aux is None:
            raise ValueError("Enhanced MOMENT must return outputs.aux_loss.")
        return mse+aux, mse, aux

    def validate():
        rng_state = capture_rng(torch)
        was_training = model.training
        total, count = 0., 0
        try:
            model.eval()
            with torch.no_grad():
                for index, (x,) in enumerate(loader(data["validation"])):
                    limit = int(train_config.get("max_validation_batches", 0))
                    if limit > 0 and index >= limit:
                        break
                    x = x.to(device)
                    with autocast():
                        output = model(x_enc=x, input_mask=torch.ones(x.shape[0], x.shape[-1], device=device))
                        loss, _, _ = losses(output, x)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite validation loss.")
                    total += float(loss)*len(x)
                    count += len(x)
        finally:
            model.train(was_training)
            restore_rng(rng_state, torch)
        return total/count

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir/"config.resolved.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
    np.savez(output_dir/"train_transforms.npz", mean=data["mean"], std=data["std"])

    def checkpoint(path, *, bank_ready=False):
        state = {"format_version": 1, "config": config, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
            "step": step, "epoch": epoch, "batch_cursor": batch_cursor,
            "consecutive_overflows": consecutive_overflows,
            "rng": capture_rng(torch), "data_source": source, "data_fingerprint": fingerprint,
            "transforms": {"mean": data["mean"], "std": data["std"]},
            "split_bounds": data["split_bounds"], "bank_ready": bank_ready,
            "resume_semantics": "next batch in deterministic seed+epoch permutation; num_workers=0",
            "runtime": {"python": sys.version, "torch": torch.__version__, "device": str(device)}}
        temporary = path.with_suffix(path.suffix+".tmp")
        torch.save(state, temporary)
        temporary.replace(path)

    with (output_dir/"metrics.jsonl").open("a", encoding="utf-8") as metrics:
        log_event(metrics, "start", device=str(device), amp=use_amp, step=step, epoch=epoch,
            batch_cursor=batch_cursor, train_windows=len(data["train"]),
            validation_windows=len(data["validation"]), test_windows=len(data["test"]),
            data_fingerprint=fingerprint, build_memory_only=args.build_memory_only)
        target_steps = int(train_config["max_steps"])
        max_overflows = int(train_config.get("max_consecutive_overflows", 100))
        train_loader_length = math.ceil(len(data["train"])/batch_size)
        if not 0 <= batch_cursor < train_loader_length:
            raise ValueError("Saved batch cursor is outside the training epoch.")
        model.train()
        while step < target_steps and not args.build_memory_only:
            order = epoch_order(len(data["train"]), seed, epoch)
            for batch_index, (x,) in enumerate(loader(data["train"], order)):
                if batch_index < batch_cursor:
                    continue
                started = time.monotonic()
                x = x.to(device)
                optimizer.zero_grad(set_to_none=True)
                with autocast():
                    output = model(x_enc=x, input_mask=torch.ones(x.shape[0], x.shape[-1], device=device))
                    loss, mse, aux = losses(output, x)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite training loss at step {step}.")
                update = optimizer_update(loss, model.parameters(), optimizer, scaler,
                    float(train_config.get("max_grad_norm", 1.)))
                updated = update["optimizer_updated"]
                batch_cursor = batch_index+1
                if batch_cursor == train_loader_length:
                    epoch += 1
                    batch_cursor = 0
                if updated:
                    step += 1
                    consecutive_overflows = 0
                else:
                    consecutive_overflows += 1
                fields = {"step": step, "epoch": epoch, "batch_cursor": batch_cursor,
                          "loss": float(loss.detach()), "reconstruction_loss": float(mse.detach()),
                          "aux_loss": float(aux.detach()), **update,
                          "consecutive_overflows": consecutive_overflows,
                          "seconds": time.monotonic()-started}
                for name in ("moe_aux_loss", "vq_loss", "vq_perplexity"):
                    value = getattr(output, name, None)
                    if value is not None:
                        fields[name] = float(value.detach())
                if updated and int(train_config.get("validation_every", 100)) > 0 and step % int(train_config.get("validation_every", 100)) == 0:
                    fields["validation_loss"] = validate()
                log_event(metrics, "train" if updated else "amp_overflow", **fields)
                if consecutive_overflows >= max_overflows:
                    checkpoint(output_dir/"checkpoint_latest.pt")
                    raise FloatingPointError(f"Stopped after {consecutive_overflows} consecutive "
                        "AMP overflows without an optimizer update; inspect data/model numerics.")
                if updated and step % int(train_config.get("checkpoint_every", 100)) == 0:
                    checkpoint(output_dir/"checkpoint_latest.pt")
                if step >= target_steps or batch_cursor == 0:
                    break
        checkpoint(output_dir/"checkpoint_latest.pt")

        # The bank is rebuilt using final weights over EVERY training window.
        # Validation/test never enter memory collection, even during standalone rebuild.
        if getattr(model, "vq_vae", None) is None:
            checkpoint(output_dir/"checkpoint_final.pt", bank_ready=False)
            log_event(metrics, "complete", step=step, epoch=epoch, memory_windows=0,
                      active_codes=0, bank_ready=False,
                      checkpoint=str(output_dir/"checkpoint_final.pt"), memory=None)
            return 0
        model.eval()
        model.vq_vae.begin_memory_collection()
        memory_windows = 0
        try:
            with torch.no_grad():
                for (x,) in loader(data["train"]):
                    x = x.to(device)
                    model.reconstruct(x_enc=x, input_mask=torch.ones(x.shape[0], x.shape[-1], device=device))
                    memory_windows += len(x)
        finally:
            model.vq_vae.end_memory_collection()
        model.vq_vae.export_memory(output_dir/"codebook_memory.pt")
        checkpoint(output_dir/"checkpoint_final.pt", bank_ready=True)
        checkpoint(output_dir/"checkpoint_latest.pt", bank_ready=True)
        log_event(metrics, "complete", step=step, epoch=epoch, memory_windows=memory_windows,
                  active_codes=int(model.vq_vae.memory_indices.numel()),
                  checkpoint=str(output_dir/"checkpoint_final.pt"),
                  memory=str(output_dir/"codebook_memory.pt"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
