from types import SimpleNamespace
import json
from pathlib import Path
import pytest
import torch
from torch import nn
from PIL import Image
from utils.options import get_args
from model.build import build_model
from model.clip_model import Transformer
from model.spa_lora import SPALoRALinear
from model.missing_aware_encoding import ModalityDropout
from model.reliability_adaptive_fusion import ReliabilityAdaptiveFusion
from utils.protocol import deterministic_masks
from utils.metrics import rank, Evaluator
from utils.checkpoint import Checkpointer
from datasets.person_data import PersonDataset, TrainDataset, QueryDataset
from datasets.build import build_transforms

torch.set_num_threads(1)
# This VM does not expose the CPU ISA required by the installed oneDNN backend.
torch.backends.mkldnn.enabled = False


def tiny_model(dataset="ORBench", add_lora=True):
    args = get_args([])
    args.add_lora = add_lora
    args.img_size = [6, 4]
    args.text_length = 5
    args.lora_r = 2
    args.spa_experts = 2
    args.spa_parts = 3
    args.lora_layers = 1
    args.completion_num_heads = 2
    args.completion_num_layers = 1
    args.completion_dropout = 0.0
    args.reliability_hidden_dim = 8
    args.dataset_name = dataset
    visual = nn.Module()
    visual.conv1 = nn.Conv2d(3, 8, 2, 2, bias=False)
    visual.class_embedding = nn.Parameter(torch.randn(8))
    visual.positional_embedding = nn.Parameter(torch.randn(7, 8))
    visual.ln_pre = nn.LayerNorm(8)
    visual.ln_post = nn.LayerNorm(8)
    visual.proj = nn.Parameter(torch.randn(8, 8) * 0.1)
    visual.transformer = Transformer(8, 2, 2)
    base = SimpleNamespace(
        visual=visual,
        token_embedding=nn.Embedding(100, 8),
        positional_embedding=nn.Parameter(torch.randn(5, 8)),
        transformer=Transformer(8, 1, 2),
        ln_final=nn.LayerNorm(8),
        text_projection=nn.Parameter(torch.randn(8, 8) * 0.1),
    )
    return build_model(args, 3, (base, {"embed_dim": 8}))


def batch():
    return {
        **{key: torch.randn(3, 3, 6, 4) for key in ("rgbs", "nirs", "cps", "sks")},
        "caption_ids": torch.tensor(
            [[1, 5, 99, 0, 0], [1, 6, 4, 99, 0], [1, 3, 2, 8, 99]]
        ),
        "pids": torch.tensor([0, 1, 2]),
        "modality_available": torch.ones(3, 5, dtype=torch.bool),
    }


def test_dropout_preserves_real_modalities_per_sample():
    available = torch.tensor(
        [[1, 0, 0, 1, 1], [1, 0, 0, 0, 0], [0, 0, 0, 1, 0]], dtype=torch.bool
    ).repeat(100, 1)
    dropout = ModalityDropout(min_keep=2, dropout_prob=1.0, keep_rgb_prob=0.0)
    mask = dropout(available)
    assert mask.any(1).all() and not (mask & ~available).any()
    assert (mask.sum(1) == available.sum(1).clamp(max=2)).all()
    assert len(mask.unique(dim=0)) >= 3
    with pytest.raises(ValueError):
        dropout(torch.zeros(2, 5, dtype=torch.bool))


def test_spa_spatial_routing_and_frozen_base():
    layer = SPALoRALinear(nn.Linear(4, 4, bias=False), rank=2, experts=2, parts=2)
    assert not layer.base.weight.requires_grad
    with torch.no_grad():
        layer.lora_A.fill_(1)
        layer.lora_B.zero_()
        layer.lora_B[0, 0].fill_(1)
        layer.lora_B[1, 1].fill_(2)
        layer.lora_router.zero_()
        layer.lora_router[:, 0, 0] = 10
        layer.lora_router[:, 1, 4] = 10
    x = torch.ones(2, 9, 4, requires_grad=True)
    state = torch.tensor([[1, 0, 0, 0, 0], [0, 0, 0, 0, 1]], dtype=torch.float)
    out = layer(x, state, (4, 2))
    delta = out - layer.base(x)
    assert not torch.allclose(out[0], out[1])
    assert not torch.allclose(delta[0, 1:5], delta[0, 5:])
    out.sum().backward()
    assert layer.lora_router.grad.abs().sum() > 0
    assert layer.base.weight.grad is None


def test_retrieval_loss_trains_completion_and_fusion():
    model = tiny_model().train()
    data = batch()
    data["modality_mask"] = torch.tensor(
        [[0, 0, 0, 0, 1], [0, 0, 0, 1, 0], [0, 1, 0, 0, 0]], dtype=torch.bool
    )
    # Neither auxiliary CMC loss is allowed to supply generator gradients here.
    model.args.completion_recon_loss_weight = (
        model.args.completion_cycle_loss_weight
    ) = 0.0
    ret = model(data, current_epoch=6)
    (ret["id_loss"] + ret["sdm_loss"]).backward()
    generator_grad = sum(
        float(p.grad.abs().sum())
        for p in model.completion_module.parameters()
        if p.grad is not None
    )
    reliability_grad = sum(
        float(p.grad.abs().sum())
        for p in model.reliability_fusion.reliability_networks.parameters()
        if p.grad is not None
    )
    assert generator_grad > 0 and reliability_grad > 0
    assert model.reliability_fusion.output_proj.weight.grad.abs().sum() > 0
    assert torch.isfinite(ret["features"]).all()
    assert torch.allclose(ret["fusion_weights"].sum(1), torch.ones(3))


def test_only_observed_inputs_affect_inference():
    model = tiny_model().eval()
    data = batch()
    available = torch.ones(3, 5, dtype=torch.bool)
    mask = torch.tensor([[0, 0, 0, 0, 1]] * 3, dtype=torch.bool)
    first = model.encode_multimodal(data, mask, available)
    for key in ("rgbs", "nirs", "cps", "sks"):
        data[key].fill_(1e6)
    second = model.encode_multimodal(data, mask, available)
    assert torch.allclose(first, second)
    with pytest.raises(ValueError):
        model.encode_multimodal(data, torch.zeros_like(mask), available)


def test_three_modal_never_trains_placeholder_targets():
    model = tiny_model("CUHK-PEDES").train()
    data = batch()
    data["modality_available"][:, 1:3] = False
    data["modality_mask"] = torch.tensor(
        [[1, 0, 0, 0, 1], [0, 0, 0, 1, 1], [1, 0, 0, 1, 0]], dtype=torch.bool
    )
    ret = model(data, current_epoch=6)
    assert ret["fusion_weights"][:, 1:3].eq(0).all()
    assert not ret["generated_mask"][:, 1:3].any()
    sum(v for k, v in ret.items() if k.endswith("_loss")).backward()
    assert model.completion_module.generators["NIR"].query_token.grad is None
    assert model.completion_module.generators["CP"].query_token.grad is None
    assert all(torch.isfinite(v) for k, v in ret.items() if k.endswith("_loss"))


def test_raf_weights_control_output_and_exact_uncertainty():
    module = ReliabilityAdaptiveFusion(8, hidden_dim=8)
    features = {"RGB": torch.ones(2, 8), "TEXT": -torch.ones(2, 8)}
    valid = torch.tensor([[1, 0, 0, 0, 1]] * 2, dtype=torch.bool)
    generated = torch.tensor([[0, 0, 0, 0, 1]] * 2, dtype=torch.bool)
    first, info = module(features, valid, generated)
    manual = (
        (
            info["fusion_weights"]
            * (generated.float() + torch.relu(0.5 - info["reliability_scores"]))
        )
        .sum(1)
        .mean()
    )
    assert torch.allclose(manual, info["uncertainty_loss"])
    with torch.no_grad():
        module.modality_bias[0] += 20
    second, _ = module(features, valid, generated)
    assert not torch.allclose(first, second)


def test_protocol_is_batch_order_independent_and_nonempty():
    available = torch.tensor(
        [[1, 0, 0, 1, 1], [1, 0, 0, 0, 0], [0, 1, 1, 1, 1]], dtype=torch.bool
    )
    ids = torch.tensor([70, 10, 99])
    full = deterministic_masks(available, ids, 0.5, 42)
    order = torch.tensor([2, 0, 1])
    permuted = deterministic_masks(available[order], ids[order], 0.5, 42)
    assert torch.equal(full[order], permuted)
    assert torch.equal(full[1], available[1])
    all_drop = deterministic_masks(available, ids, 1.0, 42)
    assert all_drop.sum(1).eq(1).all()
    assert not (all_drop & ~available).any()
    with pytest.raises(ValueError):
        deterministic_masks(available, ids, 1.1, 42)


def test_rank_handles_small_gallery_and_unmatched_queries():
    cmc, ap, inp, _ = rank(
        torch.tensor([[0.9, 0.1], [0.2, 0.8], [0.1, 0.2]]),
        torch.tensor([1, 2, 9]),
        torch.tensor([1, 2]),
    )
    assert len(cmc) == 10 and cmc.eq(100).all()
    assert ap == 100 and inp == 100
    _, ap, inp, _ = rank(
        torch.eye(2), torch.tensor([1, 2]), torch.tensor([1, 2]), get_mAP=False
    )
    assert ap is None and inp is None


def make_dataset(root, with_split=True, missing=False):
    folder = root / "CUHK-PEDES"
    (folder / "imgs" / "cam_a").mkdir(parents=True)
    skroot = root / "sketch" / "aliyun" / "CUHK" / "imgs" / "cam_a"
    skroot.mkdir(parents=True)
    rows = []
    for i, split in enumerate(("train", "val", "test"), 1):
        rgb = folder / "imgs" / "cam_a" / f"{i}.bmp"
        Image.new("RGB", (4, 6), "blue").save(rgb)
        if not (missing and i == 1):
            Image.new("RGB", (4, 6), "red").save(skroot / f"{i}.jpg")
        row = {"id": i, "file_path": f"cam_a/{i}.bmp", "captions": ["a person"]}
        if with_split:
            row["split"] = split
        rows.append(row)
    (folder / "caption_all.json").write_text(json.dumps(rows), encoding="utf-8")


def test_dataset_extension_matching_and_explicit_splits(tmp_path):
    make_dataset(tmp_path)
    ds = PersonDataset(tmp_path, "CUHK-PEDES")
    assert ds.train[0][5].endswith(".jpg") and ds.val is not None
    image = Image.open(ds.train[0][5]).convert("RGB")
    assert image.getpixel((0, 0))[0] > 200
    sample = TrainDataset(ds.train, build_transforms((6, 4)), 5)[0]
    assert sample["modality_available"].tolist() == [True, False, False, True, True]
    assert sample["nirs"].eq(0).all() and sample["cps"].eq(0).all()


def test_missing_sketch_is_masked_or_errors(tmp_path):
    make_dataset(tmp_path, missing=True)
    ds = PersonDataset(tmp_path, "CUHK-PEDES")
    assert ds.train[0][5] is None
    with pytest.raises(FileNotFoundError):
        PersonDataset(tmp_path, "CUHK-PEDES", missing_sketch_policy="error")


def test_cuhk_missing_official_split_rejected(tmp_path):
    make_dataset(tmp_path, with_split=False)
    with pytest.raises(ValueError, match="official split"):
        PersonDataset(tmp_path, "CUHK-PEDES")


def test_evaluator_and_model_use_same_encoder(tmp_path):
    model = tiny_model().eval()
    data = batch()
    data["sample_index"] = torch.arange(3)
    args = get_args([])
    args.output_dir = str(tmp_path)
    args.protocol_dir = str(tmp_path)
    evaluator = Evaluator([data], args=args)
    ids, features = evaluator.extract(model, [data], "fixture", 0.0)
    assert torch.allclose(features, model.encode_multimodal(data), atol=1e-6)
    assert len(list(tmp_path.glob("*weights.json"))) == 1


def test_checkpoint_roundtrip_and_legacy_rejection(tmp_path):
    model = tiny_model()
    c = Checkpointer(model, save_dir=str(tmp_path))
    c.save("sample", epoch=2)
    assert c.load(tmp_path / "sample.pth")["num_classes"] == 3
    torch.save({"model": model.state_dict()}, tmp_path / "legacy.pth")
    with pytest.raises(ValueError, match="not MFCRAF v2"):
        c.load(tmp_path / "legacy.pth")


@pytest.mark.parametrize(
    "off",
    [
        "use_missing_aware",
        "use_cross_modal_completion",
        "use_reliability_fusion",
        "add_lora",
    ],
)
def test_ablation_paths_run(off):
    model = tiny_model(add_lora=off != "add_lora").eval()
    if off == "add_lora":
        assert not get_args(["--no-add_lora"]).add_lora
    else:
        setattr(model, off, False)
    x = model.encode_multimodal(batch())
    assert x.shape == (3, 8) and torch.isfinite(x).all()


def test_parser_defaults_and_boolean_negation():
    a = get_args(["--no-use_missing_aware", "--no-compute_mAP"])
    assert not a.use_missing_aware and not a.compute_mAP
    assert a.add_lora and a.num_epoch == 30 and a.spa_experts == 4 and a.spa_parts == 3


def test_missing_text_content_and_length_cannot_leak():
    model = tiny_model().eval()
    data = batch()
    mask = torch.tensor([[1, 0, 0, 0, 0]] * 3, dtype=torch.bool)
    first = model.encode_multimodal(data, mask)
    data["caption_ids"] = torch.tensor([[99, 0, 0, 0, 0]] * 3)
    second = model.encode_multimodal(data, mask)
    assert torch.allclose(first, second)


def test_missing_sequence_priors_receive_retrieval_gradient():
    model = tiny_model().train()
    data = batch()
    data["modality_mask"] = torch.tensor([[0, 0, 0, 0, 1]] * 3, dtype=torch.bool)
    model.args.completion_recon_loss_weight = (
        model.args.completion_cycle_loss_weight
    ) = 0.0
    result = model(data, current_epoch=6)
    (result["id_loss"] + result["sdm_loss"]).backward()
    for tokens in model.missing_aware_encoder.missing_tokens:
        assert tokens.grad is not None and tokens.grad.abs().sum() > 0
    assert model.missing_aware_encoder.mask_embedding.weight.grad[0].abs().sum() > 0


def test_chunked_ranking_matches_full_ranking():
    from utils.metrics import score_features

    q = torch.randn(7, 8)
    g = torch.randn(11, 8)
    qids = torch.tensor([1, 2, 3, 1, 2, 3, 4])
    gids = torch.tensor([1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3])
    expected = rank(q @ g.T, qids, gids)
    actual = score_features(q, g, qids, gids, chunk_size=2)
    for a, b in zip(expected[:3], actual):
        assert torch.allclose(torch.as_tensor(a), torch.as_tensor(b), atol=1e-5)


def test_real_clip_class_with_spa_forward_backward():
    from model.clip_model import CLIP

    args = get_args([])
    args.img_size = [24, 8]
    args.stride_size = 4
    args.text_length = 7
    args.lora_layers = 1
    args.spa_parts = 3
    args.completion_num_heads = 2
    args.completion_num_layers = 1
    args.reliability_hidden_dim = 16
    args.completion_cycle_loss_weight = 0.0
    native = CLIP(
        embed_dim=16,
        image_resolution=(24, 8),
        vision_layers=2,
        vision_width=64,
        vision_patch_size=4,
        stride_size=4,
        context_length=7,
        vocab_size=100,
        transformer_width=64,
        transformer_heads=1,
        transformer_layers=2,
    )
    model = build_model(args, 2, (native, {"embed_dim": 16}))
    data = {
        **{key: torch.randn(2, 3, 24, 8) for key in ("rgbs", "nirs", "cps", "sks")},
        "caption_ids": torch.tensor([[1, 3, 99, 0, 0, 0, 0], [1, 6, 5, 99, 0, 0, 0]]),
        "pids": torch.tensor([0, 1]),
        "modality_mask": torch.tensor(
            [[0, 0, 0, 1, 1], [0, 1, 0, 0, 0]], dtype=torch.bool
        ),
    }
    output = model(data, current_epoch=6)
    sum(x for k, x in output.items() if k.endswith("_loss")).backward()
    assert output["features"].shape == (2, 16)
    assert not model.visual.conv1.weight.requires_grad


def test_train_loop_uses_effective_losses_and_updates_parameters():
    from train import train_one_epoch
    from solver import build_optimizer, build_lr_scheduler

    model = tiny_model()
    args = model.args
    args.amp = False
    data = batch()
    data["modality_mask"] = torch.tensor([[0, 0, 0, 1, 1]] * 3, dtype=torch.bool)
    optimizer = build_optimizer(args, model)
    scheduler = build_lr_scheduler(args, optimizer)
    before = model.reliability_fusion.output_proj.weight.detach().clone()
    stats = train_one_epoch(args, model, [data], optimizer, scheduler, 6)
    assert stats and not torch.equal(
        before, model.reliability_fusion.output_proj.weight
    )
    assert all(torch.isfinite(torch.tensor(x)) for x in stats.values())


def test_supplied_zip_rgb_sketch_pair_smoke(tmp_path):
    from zipfile import ZipFile

    root = Path(__file__).resolve().parents[1]
    rgbzip = root / "data/CUHK-PEDES.zip"
    skzip = root / "data/Tri-CUHK-PEDES, Tri-ICFG-PEDES, and Tri-RSTPReid.zip"
    if not rgbzip.exists() or not skzip.exists():
        pytest.skip("Optional real-data ZIP fixtures absent")
    with ZipFile(rgbzip) as z, ZipFile(skzip) as sz:
        rows = json.loads(z.read("CUHK-PEDES/caption_all.json"))
        selected = []
        for row in rows:
            if row["file_path"].endswith(".bmp") and row["id"] not in {
                x["id"] for x in selected
            }:
                selected.append(dict(row))
                if len(selected) == 3:
                    break
        for row, split in zip(selected, ("train", "val", "test")):
            # Fixture-only partitions validate the reader; they are NOT a benchmark split.
            row["split"] = split
            relative = row["file_path"]
            target = tmp_path / "CUHK-PEDES" / "imgs" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read("CUHK-PEDES/imgs/" + relative))
            skrelative = str(Path(relative).with_suffix(".jpg")).replace(chr(92), "/")
            sktarget = tmp_path / "sketch" / "aliyun" / "CUHK" / "imgs" / skrelative
            sktarget.parent.mkdir(parents=True, exist_ok=True)
            sktarget.write_bytes(sz.read("sketch/aliyun/CUHK/imgs/" + skrelative))
        annotation = tmp_path / "CUHK-PEDES" / "caption_all.json"
        annotation.write_text(json.dumps(selected), encoding="utf-8")
    ds = PersonDataset(tmp_path, "CUHK-PEDES")
    sample = TrainDataset(ds.train, build_transforms((24, 8)), 7)[0]
    assert sample["sks"].shape == (3, 24, 8) and sample["modality_available"][3]
    assert not ds.missing_sketches


def test_config_file_cli_override_and_metadata(tmp_path):
    import yaml

    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump({"num_epoch": 40, "num_classes": 7, "use_missing_aware": False}),
        encoding="utf-8",
    )
    args = get_args(
        ["--config_file", str(path), "--num_epoch", "35", "--use_missing_aware"]
    )
    assert args.num_epoch == 35 and args.num_classes == 7 and args.use_missing_aware


def test_native_checkpoint_training_evaluation_and_export_cli(tmp_path):
    import subprocess
    import sys
    import yaml
    from model.clip_model import CLIP

    root = Path(__file__).resolve().parents[1]
    data_root = tmp_path / "data"
    make_dataset(data_root)
    native = CLIP(
        embed_dim=16,
        image_resolution=(24, 8),
        vision_layers=2,
        vision_width=64,
        vision_patch_size=4,
        stride_size=4,
        context_length=7,
        vocab_size=49408,
        transformer_width=64,
        transformer_heads=1,
        transformer_layers=2,
    )
    pretrained = tmp_path / "random_clip_fixture.pth"
    torch.save(native.state_dict(), pretrained)
    args = get_args([])
    config = vars(args)
    config.update(
        dataset_name="CUHK-PEDES",
        root_dir=str(data_root),
        output_dir=str(tmp_path / "run"),
        pretrain_choice=str(pretrained),
        img_size=[24, 8],
        stride_size=4,
        text_length=7,
        lora_layers=1,
        spa_experts=2,
        spa_parts=3,
        completion_num_heads=2,
        completion_num_layers=1,
        reliability_hidden_dim=16,
        completion_cycle_loss_weight=0.0,
        num_epoch=1,
        warmup_epochs=0,
        modality_dropout_warmup_epochs=0,
        completion_warmup_epochs=0,
        fusion_warmup_epochs=0,
        test_batch_size=2,
        device="cpu",
        disable_mkldnn=True,
        cpu_threads=1,
    )
    cfg = tmp_path / "train.yaml"
    cfg.write_text(yaml.safe_dump(config), encoding="utf-8")
    commands = [
        [str(root / "train.py"), "--config_file", str(cfg)],
        [
            str(root / "test.py"),
            "--config_file",
            str(tmp_path / "run/configs.yaml"),
            "--missing_rates",
            "0",
            "0.5",
            "--query_tasks",
            "TEXT+SK",
            "--device",
            "cpu",
        ],
        [
            str(root / "infer.py"),
            "--config_file",
            str(tmp_path / "run/configs.yaml"),
            "--sk_path",
            str(data_root / "sketch/aliyun/CUHK/imgs/cam_a/3.jpg"),
            "--text",
            "a person",
            "--device",
            "cpu",
            "--output",
            str(tmp_path / "retrieval.png"),
        ],
        [
            str(root / "tools/export_features.py"),
            "--config_file",
            str(tmp_path / "run/configs.yaml"),
            "--task",
            "TEXT+SK",
            "--mask_modalities",
            "SK",
            "--missing_rate",
            "0",
            "--max_samples",
            "1",
            "--device",
            "cpu",
            "--output",
            str(tmp_path / "features.npz"),
        ],
    ]
    for command in commands:
        done = subprocess.run(
            [sys.executable, *command],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert done.returncode == 0, done.stdout + "\n" + done.stderr
    assert (tmp_path / "run/best.pth").exists()
    assert (tmp_path / "retrieval.png").exists()
    assert (tmp_path / "run/run_manifest.json").exists()
    metrics = json.loads(
        (tmp_path / "run/evaluation/missing_0.5_seed_42/metrics.json").read_text()
    )
    assert "TEXT+SK" in metrics["results"] and metrics["checkpoint_sha256"]
    import numpy as np

    with np.load(tmp_path / "features.npz") as features:
        assert features["SK_real"].shape == features["SK_completed"].shape == (1, 16)


def test_build_loader_uses_validation_and_multimodal_gallery(tmp_path):
    from datasets import build_dataloader

    make_dataset(tmp_path)
    args = get_args([])
    args.root_dir = str(tmp_path)
    args.dataset_name = "CUHK-PEDES"
    args.img_size = [6, 4]
    args.text_length = 5
    loaders = build_dataloader(args)
    assert loaders[1].dataset.samples[0][0] == 2  # Official val, not test identity 3.
    args.training = False
    manifest = tmp_path / "gallery.json"
    manifest.write_text(
        json.dumps(
            {
                "modalities": ["RGB", "TEXT"],
                "samples": [[3, "CUHK-PEDES/imgs/cam_a/3.bmp", "a person"]],
            }
        ),
        encoding="utf-8",
    )
    args.gallery_file = str(manifest)
    gallery = next(iter(build_dataloader(args)[0]))
    assert gallery["modality_available"].tolist() == [[True, False, False, False, True]]
    args.training = True
    args.val_dataset = "test"
    with pytest.raises(ValueError, match="cannot select"):
        build_dataloader(args)


def test_clip_tokenizer_keeps_original_vocabulary():
    from utils.simple_tokenizer import SimpleTokenizer

    tokenizer = SimpleTokenizer()
    assert len(tokenizer.encoder) == 49408
    assert tokenizer.encoder["<|startoftext|>"] == 49406
    assert tokenizer.encoder["<|endoftext|>"] == 49407
    assert "<|mask|>" not in tokenizer.encoder


def test_reconstruction_averages_each_sample_then_the_full_batch():
    from model.cross_modal_completion import CrossModalCompletionModule

    completion = CrossModalCompletionModule(2, num_heads=1, num_layers=1)
    rgb = torch.tensor([[0.0, 1.0], [-1.0, 0.0], [1.0, 0.0]], requires_grad=True)
    predictions = {"RGB": rgb, "NIR": torch.tensor([[1.0, 0.0]] * 3)}
    targets = {
        "RGB": torch.tensor([[1.0, 0.0]] * 3),
        "NIR": torch.tensor([[1.0, 0.0]] * 3),
    }
    masked = torch.tensor(
        [[1, 0, 0, 0, 0], [1, 1, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=torch.bool
    )
    # Sample losses are 1, (2 + 0)/2, and 0: the paper's batch mean is 2/3.
    loss = completion.reconstruction_loss(predictions, targets, masked)
    assert float(loss.detach()) == pytest.approx(2 / 3)
    loss.backward()
    assert rgb.grad[0].abs().sum() > 0
    assert rgb.grad[2].eq(0).all()


def test_cycle_uses_per_sample_pair_counts_and_full_batch_mean():
    from model.cross_modal_completion import CrossModalCompletionModule, MODALITIES

    class ConstantGenerator(nn.Module):
        def __init__(self, value):
            super().__init__()
            self.register_buffer("value", torch.tensor(value, dtype=torch.float))

        def forward(self, context, context_mask, query_prior=None):
            return self.value.expand(len(context), -1)

    completion = CrossModalCompletionModule(2, num_heads=1, num_layers=1)
    for modality in MODALITIES:
        completion.modality_context_proj[modality] = nn.Identity()
        completion.generators[modality] = ConstantGenerator(
            [0.0, 1.0] if modality == "NIR" else [1.0, 0.0]
        )
    features = {m: torch.tensor([[1.0, 0.0]] * 3) for m in MODALITIES}
    features["RGB"][1] = torch.tensor([-1.0, 0.0])
    features["NIR"][:] = torch.tensor([0.0, 1.0])
    available = torch.tensor(
        [[1, 1, 0, 0, 0], [1, 1, 1, 0, 0], [1, 0, 0, 0, 0]], dtype=torch.bool
    )
    active = torch.tensor([1, 1, 1, 0, 0], dtype=torch.bool)
    # Sample 0 has two perfect cycles. Sample 1 has six cycles, two with loss 2.
    # Sample 2 has no valid pair and contributes zero: (0 + 4/6 + 0)/3 = 2/9.
    loss = completion.cycle_loss(features, available, active, max_pairs=20)
    assert float(loss) == pytest.approx(2 / 9)


def test_epoch_sampling_restores_captions_and_same_identity_pairs():
    import random

    rgb = "dataset/vis/0007/rgb.jpg"
    pools = {m: {"0007": [m + "_a", m + "_b"]} for m in ("nir", "cp", "sk")}
    records = [(0, 0, rgb, "old_nir", "old_cp", "old_sk", "caption_a")]
    dataset = TrainDataset(
        records,
        None,
        caption_choices={rgb: ["caption_a", "caption_b"]},
        modality_pools=pools,
    )
    random.seed(9)
    dataset.random_sampling()
    first = dataset.dataset[0]
    random.seed(9)
    dataset.random_sampling()
    assert dataset.dataset[0] == first
    seen = set()
    for seed in range(20):
        random.seed(seed)
        dataset.random_sampling()
        row = dataset.dataset[0]
        seen.add(row[6])
        assert row[:3] == records[0][:3]
        for column, modality in enumerate(("nir", "cp", "sk"), 3):
            assert row[column] in pools[modality]["0007"]
    assert seen == {"caption_a", "caption_b"}
    assert records[0][3] == "old_nir"  # No mutation of source annotations.


def test_training_epoch_calls_dataset_resampling():
    from train import train_one_epoch
    from solver import build_optimizer, build_lr_scheduler

    class Loader(list):
        def __init__(self):
            super().__init__([batch()])
            self.calls = 0
            self.dataset = SimpleNamespace(random_sampling=self.resample)

        def resample(self):
            self.calls += 1

    loader = Loader()
    model = tiny_model()
    model.args.completion_cycle_loss_weight = 0.0
    optimizer = build_optimizer(model.args, model)
    scheduler = build_lr_scheduler(model.args, optimizer)
    train_one_epoch(model.args, model, loader, optimizer, scheduler, 6)
    assert loader.calls == 1
