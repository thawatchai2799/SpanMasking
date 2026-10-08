"""
p1_experiment.py -- canonical runner for the span-masking JEPA experiment.

Every number that reaches the paper is produced by this file. No other script
may write results. See experiment_spec.md for the design this implements.

Subcommands
-----------
  selftest   validate mask sampler, ledger, and this file's own portability
             (numpy only; torch is NOT required for this step)
  time       200-step timing run, then recompute and print the ledger
  train      train one run:  --run-id P1_B_seed0
  probe      probe one finished run against the frozen checkpoints
  all        execute the full ledger in order, skipping finished runs
  aggregate  collect all results/*.json into results/summary.csv

Typical use on the target machine (Windows 11, one GPU):
  python p1_experiment.py selftest
  python p1_experiment.py time
  python p1_experiment.py all

All paths are handled with pathlib and are safe on Windows. All console
output is plain ASCII. Interrupting and re-running "all" resumes where it
stopped: a run is skipped iff results/<run_id>.json exists.
"""

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass, asdict, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
RESULTS = ROOT / "results"
DATA = ROOT / "data"

# ---------------------------------------------------------------- config

# cell -> (k, l) at T = 128.
#   A, B, D are the pre-registered P1 cells.
#   E, F, G extend B and D into a five-point curve at a FIXED masking
#   ratio rho = 0.25 (m = k*l = 32 for all of E, B, D, F, G), so the only
#   quantity that varies along the curve is the span length:
#       l =  1 (E, k=32),  2 (F, k=16),  4 (B, k=8),
#            8 (D, k=4),  16 (G, k=2)
# Identity stamp. The file name stays constant on purpose -- this script
# is meant to be dropped in place, and resume works off results/ rather
# than the file name -- so the version is printed instead, by selftest and
# at the top of every run. If the banner does not say P1b, the old file is
# still the one being executed.
SCRIPT_VERSION = ("2026-09-27 P1d  (156-run ledger: 34 pre-registered + 15 "
                   "P1b + 92 power extension + 15 reviewer-requested pure "
                   "span-MLM baseline at cell B)")

CELLS = {"A": (8, 2), "B": (8, 4), "D": (4, 8),
         "E": (32, 1), "F": (16, 2), "G": (2, 16)}


@dataclass
class Config:
    # model
    d_model: int = 192
    n_layers: int = 4
    n_heads: int = 4
    d_ffn: int = 512
    seq_len: int = 128
    vocab: int = 8192          # BPE merges; +1 mask token appended internally
    # training
    batch: int = 64
    steps: int = 9000
    lr: float = 3e-4
    weight_decay: float = 0.04
    warmup: int = 500
    tau: float = 0.996         # EMA for target encoder
    clip: float = 1.0          # global-norm; 0 disables
    optimizer: str = "adamw"   # "adamw" | "sgd" (P3 runs use sgd, no momentum)
    # objective arms
    arm: str = "cos"           # "cos" | "mlm" | "vicreg" | "ldb" | "mlmonly"
    # "mlmonly": conventional span-masked MLM -- cross-entropy on the masked
    # positions only, no JEPA cosine term, no target encoder. Added at a
    # reviewer's request as a comparison baseline (R2, review round 1).
    mlm_alpha: float = 0.1
    vic_mu: float = 25.0
    vic_nu: float = 1.0
    ldb_beta: float = 0.01
    ldb_eps: float = 1e-4
    ldb_kappa: float = 1.0
    # masking
    cell: str = "B"
    # bookkeeping
    seed: int = 0
    data: str = "wikitext"     # "wikitext" | "synthetic"
    diag_every: int = 250
    device: str = "auto"


def run_ledger():
    """The full set of runs, in execution order. Names are stable IDs."""
    runs = []
    for cell in ("A", "B", "D"):
        for s in range(5):
            runs.append((f"P1_{cell}_seed{s}", Config(cell=cell, seed=s)))
    for s in range(5):
        runs.append((f"P1_Aeq_seed{s}",
                     Config(cell="A", seed=s, steps=18000)))
    for s in range(5):
        runs.append((f"P4_Bmlm_seed{s}",
                     Config(cell="B", seed=s, arm="mlm")))
    for om in (0.04, 0.2):
        runs.append((f"P3_B_sgd_wd{om}",
                     Config(cell="B", seed=0, optimizer="sgd",
                            weight_decay=om, clip=0.0)))
    for s in range(3):
        runs.append((f"S6_Bvicreg_seed{s}",
                     Config(cell="B", seed=s, arm="vicreg")))
        runs.append((f"S6_Bldb_seed{s}",
                     Config(cell="B", seed=s, arm="ldb")))
    # P1b: the span-length curve at fixed ratio. Appended last so that a
    # partially finished campaign keeps its original execution order and
    # the 34 pre-registered runs are never re-run.
    for cell, tag in (("E", "l1"), ("F", "l2"), ("G", "l16")):
        for s in range(5):
            runs.append((f"P1b_{tag}_seed{s}", Config(cell=cell, seed=s)))
    runs.append(("BASE_randominit", Config(cell="B", seed=0, steps=0)))

    # ---- power extension (added after the 49-run campaign was frozen) ----
    # Seeds 5-14 for every cell/arm that had 5, and seeds 3-8 for the two
    # objective arms that had 3, bringing each to 3x its original count.
    # This is an extension of statistical power on the SAME pre-registered
    # comparisons, not a new hypothesis or a new cell -- P1, P1b and the
    # objective-arm decision rules are unchanged, only n grows. Appended
    # after every original-wave run (including P1b and BASE) so a
    # partially-finished original campaign is never disturbed, matching
    # the same discipline used when the P1b wave itself was added. Run IDs
    # continue the existing seed numbering rather than renumbering
    # anything, so results/<run_id>.json resume-detection needs no change.
    for cell in ("A", "B", "D"):
        for s in range(5, 15):
            runs.append((f"P1_{cell}_seed{s}", Config(cell=cell, seed=s)))
    for s in range(5, 15):
        runs.append((f"P1_Aeq_seed{s}",
                     Config(cell="A", seed=s, steps=18000)))
    for s in range(5, 15):
        runs.append((f"P4_Bmlm_seed{s}",
                     Config(cell="B", seed=s, arm="mlm")))
    for cell, tag in (("E", "l1"), ("F", "l2"), ("G", "l16")):
        for s in range(5, 15):
            runs.append((f"P1b_{tag}_seed{s}", Config(cell=cell, seed=s)))
    for s in range(3, 9):
        runs.append((f"S6_Bvicreg_seed{s}",
                     Config(cell="B", seed=s, arm="vicreg")))
        runs.append((f"S6_Bldb_seed{s}",
                     Config(cell="B", seed=s, arm="ldb")))
    # ---- review-round-1 extension (added 2026-09-27, after peer review) ----
    # Reviewer 2 asked for a conventional span-masked MLM baseline. Same
    # encoder, cell B masking, schedule and probes; the objective is
    # cross-entropy on masked positions alone. Appended after every prior
    # run so nothing already finished is disturbed; 15 seeds to match B.
    for s in range(15):
        runs.append((f"R2_Bmlmonly_seed{s}",
                     Config(cell="B", seed=s, arm="mlmonly")))
    return runs


# ---------------------------------------------------------------- masking

def sample_mask(rng, T, k, l):
    """k non-overlapping, non-touching blocks of exact length l in [0, T).

    Stars-and-bars over the k+1 gaps: internal gaps get a mandatory +1 so
    blocks never touch (touching blocks would merge and change k).
    Returns a sorted array of masked positions, length k*l.
    """
    free = T - k * l - (k - 1)
    if free < 0:
        raise ValueError("cell does not fit: T=%d k=%d l=%d" % (T, k, l))
    # exact-uniform composition of `free` into k+1 gaps: draw k DISTINCT
    # dividers from range(free + k), sort, subtract index (standard trick;
    # sorted iid draws would bias against repeated gap sizes)
    div = np.sort(rng.choice(free + k, size=k, replace=False))
    gaps = np.diff(np.concatenate([[-1], div])) - 1
    pos = []
    start = 0
    for i in range(k):
        start = start + gaps[i] + (1 if i > 0 else 0)
        pos.extend(range(start, start + l))
        start += l
    pos = np.asarray(pos)
    assert pos.max() < T and len(pos) == k * l
    return pos


def mask_batch(rng, B, T, k, l):
    return np.stack([sample_mask(rng, T, k, l) for _ in range(B)])


# ---------------------------------------------------------------- data

def load_token_stream(cfg):
    """Return a 1-D int32 numpy array of token ids (cached on disk)."""
    DATA.mkdir(exist_ok=True)
    if cfg.data == "synthetic":
        rng = np.random.default_rng(0)
        return rng.integers(0, cfg.vocab, size=2_000_000, dtype=np.int32)
    cache = DATA / ("wt103_%d.npy" % cfg.vocab)
    if cache.exists():
        return np.load(cache, mmap_mode="r")
    from tokenizers import ByteLevelBPETokenizer   # lazy heavy imports
    txt = DATA / "wt103_train.txt"
    if not txt.exists():
        # download only when the raw-text cache is genuinely absent; the
        # previous version called load_dataset unconditionally, re-fetching
        # the dataset even when only the .npy for a new vocab was missing
        print("[data] first use: downloading WikiText-103 and training BPE.")
        print("[data] this happens once and is cached under", str(DATA))
        from datasets import load_dataset
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1",
                          split="train")
        with open(txt, "w", encoding="utf-8", newline="") as f:
            for row in ds:
                if row["text"].strip():
                    f.write(row["text"])
    tok_dir = DATA / ("bpe_%d" % cfg.vocab)
    if not (tok_dir / "vocab.json").exists():
        tok_dir.mkdir(exist_ok=True)
        tk = ByteLevelBPETokenizer()
        tk.train(files=[str(txt)], vocab_size=cfg.vocab, min_frequency=2)
        tk.save_model(str(tok_dir))
    tk = load_tokenizer(cfg)
    ids = []
    budget = 80_000_000
    with open(txt, encoding="utf-8") as f:
        for line in f:
            ids.extend(tk.encode(line).ids)
            if len(ids) >= budget:
                break
    arr = np.asarray(ids[:budget], dtype=np.int32)
    np.save(cache, arr)
    return np.load(cache, mmap_mode="r")


def load_tokenizer(cfg):
    from tokenizers import ByteLevelBPETokenizer
    tok_dir = DATA / ("bpe_%d" % cfg.vocab)
    return ByteLevelBPETokenizer(str(tok_dir / "vocab.json"),
                                 str(tok_dir / "merges.txt"))


def batch_iter(stream, cfg, seed):
    rng = np.random.default_rng(10_000 + seed)
    n = len(stream) - cfg.seq_len - 1
    while True:
        idx = rng.integers(0, n, size=cfg.batch)
        x = np.stack([np.asarray(stream[i:i + cfg.seq_len]) for i in idx])
        yield x.astype(np.int64)


# ---------------------------------------------------------------- model

def build_model(cfg, torch):
    nn = torch.nn

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(cfg.vocab + 1, cfg.d_model)  # +1 = mask
            self.pos = nn.Embedding(cfg.seq_len, cfg.d_model)
            layer = nn.TransformerEncoderLayer(
                d_model=cfg.d_model, nhead=cfg.n_heads,
                dim_feedforward=cfg.d_ffn, batch_first=True,
                norm_first=True, dropout=0.0, activation="gelu")
            self.blocks = nn.TransformerEncoder(
                layer, cfg.n_layers, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(cfg.d_model)

        def forward(self, ids, pad_mask=None):
            # pad_mask: (B, L) bool, True = PADDING position to be ignored.
            # Training always uses full-length windows (pad_mask=None);
            # probing on variable-length text must pass it, otherwise pad
            # tokens (id 0 is a real BPE symbol) contaminate attention.
            p = torch.arange(ids.shape[1], device=ids.device)
            h = self.emb(ids) + self.pos(p)[None, :, :]
            return self.norm(self.blocks(h, src_key_padding_mask=pad_mask))

    class Predictor(nn.Module):
        def __init__(self):
            super().__init__()
            self.f1 = nn.Linear(cfg.d_model, cfg.d_model)
            self.act = nn.GELU()
            self.f2 = nn.Linear(cfg.d_model, cfg.d_model)  # final linear: Thm 2

        def forward(self, h):
            return self.f2(self.act(self.f1(h)))

    return Encoder(), Predictor()


# ---------------------------------------------------------------- losses

def gather_positions(t, pos, torch):
    """t: (B, T, D); pos: (B, m) long -> (B, m, D)."""
    idx = pos.unsqueeze(-1).expand(-1, -1, t.shape[-1])
    return torch.gather(t, 1, idx)


def covariance(z):
    zc = z - z.mean(0, keepdim=True)
    return (zc.T @ zc) / (z.shape[0] - 1)


def regulariser(zflat, cfg, torch):
    """zflat: (N, D) predictor outputs at masked positions."""
    if cfg.arm == "vicreg":
        # canonical VICReg normalisation (Bardes et al.): the variance term
        # is a MEAN over dimensions and the covariance term is divided by d;
        # a sum-form would be ~d times stronger than the published objective
        C = covariance(zflat)
        d = C.shape[0]
        var = torch.sqrt(torch.clamp(torch.diagonal(C), min=0) + 1e-8)
        hinge = torch.clamp(1.0 - var, min=0.0)
        off = C - torch.diag(torch.diagonal(C))
        return (cfg.vic_mu * (hinge ** 2).mean()
                + cfg.vic_nu * (off ** 2).sum() / d)
    if cfg.arm == "ldb":
        C = covariance(zflat)
        d = C.shape[0]
        eye = torch.eye(d, device=C.device, dtype=C.dtype)
        ent = -(cfg.ldb_beta / 2.0) * torch.logdet(C + cfg.ldb_eps * eye)
        tr = torch.trace(C) / d
        return ent + (cfg.ldb_kappa / 2.0) * (tr - 1.0) ** 2
    return zflat.new_zeros(())


# ---------------------------------------------------------------- training

def get_device(cfg, torch):
    if cfg.device != "auto":
        return torch.device(cfg.device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    print("[warn] no CUDA device found; running on CPU will be very slow.")
    return torch.device("cpu")


def train_run(run_id, cfg):
    import torch
    RUNS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    out = RUNS / run_id
    out.mkdir(exist_ok=True)
    dev = get_device(cfg, torch)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    enc, pred = build_model(cfg, torch)
    enc.to(dev); pred.to(dev)
    tgt = build_model(cfg, torch)[0].to(dev)
    tgt.load_state_dict(enc.state_dict())
    for p in tgt.parameters():
        p.requires_grad_(False)
    mlm_head = torch.nn.Linear(cfg.d_model, cfg.vocab).to(dev) \
        if cfg.arm in ("mlm", "mlmonly") else None

    params = list(enc.parameters()) + list(pred.parameters())
    if mlm_head is not None:
        params += list(mlm_head.parameters())
    if cfg.optimizer == "sgd":
        opt = torch.optim.SGD(params, lr=cfg.lr, momentum=0.0,
                              weight_decay=cfg.weight_decay)
    else:
        opt = torch.optim.AdamW(params, lr=cfg.lr,
                                weight_decay=cfg.weight_decay)

    def lr_at(step):
        if step < cfg.warmup:
            return cfg.lr * (step + 1) / cfg.warmup
        t = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
        return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * min(t, 1.0)))

    stream = load_token_stream(cfg)
    batches = batch_iter(stream, cfg, cfg.seed)
    mrng = np.random.default_rng(20_000 + cfg.seed)
    k, l = CELLS[cfg.cell]
    MASK = cfg.vocab   # id of the mask token

    # fixed diagnostic batch: same sequences and same masks every time
    drng = np.random.default_rng(999)
    diag_x = next(batch_iter(stream, cfg, 999))
    diag_pos = mask_batch(drng, cfg.batch, cfg.seq_len, k, l)
    diag_log = []

    def diagnostics(step, loss_val):
        enc.eval(); pred.eval()
        with torch.no_grad():
            x = torch.as_tensor(diag_x, device=dev)
            pos = torch.as_tensor(diag_pos, device=dev)
            xc = x.clone()
            xc.scatter_(1, pos, MASK)
            z = gather_positions(enc(xc), pos, torch)
            if cfg.arm != "mlmonly":   # predictor is untrained under pure MLM
                z = pred(z)
            z = z.reshape(-1, cfg.d_model).double()
            C = covariance(z)
            ev = torch.linalg.eigvalsh(C).clamp(min=0)
            p = ev / ev.sum()
            p = p[p > 1e-12]
            er = float(torch.exp(-(p * torch.log(p)).sum()))
            wf = float(torch.linalg.norm(pred.f2.weight))
        enc.train(); pred.train()
        # lr is logged because the P3 fit must use cumulative lr as the
        # time variable: with a cosine schedule, steps are not proportional
        # to gradient-flow time
        diag_log.append(dict(step=step, loss=loss_val, lr=lr_at(step),
                             lam_min=float(ev[0]), eff_rank=er, w_frob=wf))

    t0 = time.time()
    loss_val = float("nan")
    for step in range(cfg.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        x = torch.as_tensor(next(batches), device=dev)
        pos = torch.as_tensor(
            mask_batch(mrng, cfg.batch, cfg.seq_len, k, l), device=dev)
        xc = x.clone()
        xc.scatter_(1, pos, MASK)

        hc = gather_positions(enc(xc), pos, torch)
        zh = pred(hc)
        with torch.no_grad():
            zt = gather_positions(tgt(x), pos, torch)
            zt = torch.nn.functional.normalize(zt, dim=-1)
        zh_n = torch.nn.functional.normalize(zh, dim=-1)
        loss = (1.0 - (zh_n * zt).sum(-1)).mean()
        if cfg.arm == "mlm":
            logits = mlm_head(hc)
            tgt_ids = torch.gather(x, 1, pos)
            loss = loss + cfg.mlm_alpha * torch.nn.functional.cross_entropy(
                logits.reshape(-1, cfg.vocab), tgt_ids.reshape(-1))
        elif cfg.arm == "mlmonly":
            # pure MLM: the cosine term is dropped entirely (the predictor
            # and target encoder receive no gradient and play no role)
            logits = mlm_head(hc)
            tgt_ids = torch.gather(x, 1, pos)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, cfg.vocab), tgt_ids.reshape(-1))
        loss = loss + regulariser(zh.reshape(-1, cfg.d_model), cfg, torch)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.clip > 0:
            torch.nn.utils.clip_grad_norm_(params, cfg.clip)
        opt.step()
        with torch.no_grad():
            for pt, po in zip(tgt.parameters(), enc.parameters()):
                pt.mul_(cfg.tau).add_(po, alpha=1.0 - cfg.tau)
        loss_val = float(loss.detach())
        if step % cfg.diag_every == 0:
            diagnostics(step, loss_val)
        if step % 500 == 0:
            el = time.time() - t0
            print("[%s] step %d/%d loss %.4f elapsed %.1f min"
                  % (run_id, step, cfg.steps, loss_val, el / 60.0))
    diagnostics(cfg.steps, loss_val)

    torch.save(enc.state_dict(), out / "encoder.pt")
    res = dict(run_id=run_id, config=asdict(cfg),
               minutes=(time.time() - t0) / 60.0, diagnostics=diag_log)
    with open(RESULTS / (run_id + ".json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print("[%s] done in %.1f min" % (run_id, res["minutes"]))
    return res


# ---------------------------------------------------------------- probes

def extract_features(enc, texts, cfg, torch, dev, tk):
    """Mean-pooled encoder states of UNMASKED inputs. Returns (N, D)."""
    feats = []
    enc.eval()
    with torch.no_grad():
        for i in range(0, len(texts), 128):
            chunk = texts[i:i + 128]
            ids = [tk.encode(t).ids[:cfg.seq_len] for t in chunk]
            L = max(1, max(len(s) for s in ids))
            arr = np.zeros((len(ids), L), dtype=np.int64)
            msk = np.zeros((len(ids), L), dtype=bool)
            for j, s in enumerate(ids):
                arr[j, :len(s)] = s
                msk[j, :len(s)] = True
            msk[:, 0] = True   # guard: an all-pad row would make softmax NaN
            pad = torch.as_tensor(~msk, device=dev)
            h = enc(torch.as_tensor(arr, device=dev),
                    pad_mask=pad).cpu().numpy()
            m = msk[:, :, None]
            feats.append((h * m).sum(1) / np.maximum(m.sum(1), 1))
    return np.concatenate(feats)


def probe_run(run_id):
    import torch
    res_file = RESULTS / (run_id + ".json")
    if not res_file.exists():
        sys.exit("no results for %s -- run 'train --run-id %s' first"
                 % (run_id, run_id))
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from datasets import load_dataset
    cfg_d = json.load(open(RESULTS / (run_id + ".json"),
                           encoding="utf-8"))["config"]
    cfg = Config(**cfg_d)
    dev = get_device(cfg, torch)
    enc, _ = build_model(cfg, torch)
    enc.load_state_dict(torch.load(RUNS / run_id / "encoder.pt",
                                   map_location=dev, weights_only=True))
    enc.to(dev)
    tk = load_tokenizer(cfg)

    scores = {}
    ag = load_dataset("fancyzhx/ag_news")
    rng = np.random.default_rng(0)
    sub = rng.choice(len(ag["train"]), 20000, replace=False)
    Xtr = extract_features(enc, [ag["train"][int(i)]["text"] for i in sub],
                           cfg, torch, dev, tk)
    ytr = np.asarray([ag["train"][int(i)]["label"] for i in sub])
    Xte = extract_features(enc, ag["test"]["text"], cfg, torch, dev, tk)
    yte = np.asarray(ag["test"]["label"])
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000).fit(sc.transform(Xtr), ytr)
    scores["agnews"] = float(clf.score(sc.transform(Xte), yte))

    sst = load_dataset("nyu-mll/glue", "sst2")
    Xtr = extract_features(enc, sst["train"]["sentence"], cfg, torch, dev, tk)
    ytr = np.asarray(sst["train"]["label"])
    Xte = extract_features(enc, sst["validation"]["sentence"],
                           cfg, torch, dev, tk)
    yte = np.asarray(sst["validation"]["label"])
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000).fit(sc.transform(Xtr), ytr)
    scores["sst2"] = float(clf.score(sc.transform(Xte), yte))

    scores.update(probe_lexical(enc, cfg, torch, dev, tk))

    res_path = RESULTS / (run_id + ".json")
    res = json.load(open(res_path, encoding="utf-8"))
    res["probes"] = scores
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print("[%s] probes: %s" % (run_id, scores))
    return scores


def probe_lexical(enc, cfg, torch, dev, tk):
    """Word-content and POS probes on UNMASKED text (see spec section 3).

    Word-content: contextual state at position t -> token id at t, top-1k
    vocabulary. Uses windows from the pretraining stream; the encoder has
    seen this text, which is standard for probing and identical across runs.
    POS: Brown corpus, universal tagset, first-subtoken convention.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    scores = {}
    stream = load_token_stream(cfg)
    rng = np.random.default_rng(7)
    counts = np.bincount(np.asarray(stream[:2_000_000]),
                         minlength=cfg.vocab)
    n_top = min(1000, cfg.vocab)      # top-1k in production; smaller vocabs
    top = np.argsort(counts)[::-1][:n_top]   # (e.g. smoke tests) must not crash
    rank = -np.ones(cfg.vocab, dtype=np.int64)
    rank[top] = np.arange(n_top)
    X, y = [], []
    enc.eval()
    with torch.no_grad():
        while len(y) < 25000:
            i = int(rng.integers(0, len(stream) - cfg.seq_len))
            ids = np.asarray(stream[i:i + cfg.seq_len], dtype=np.int64)
            h = enc(torch.as_tensor(ids[None, :], device=dev))[0]
            keep = np.where(rank[ids] >= 0)[0]
            if len(keep) == 0:
                continue
            keep = rng.choice(keep, size=min(8, len(keep)), replace=False)
            X.append(h[keep].cpu().numpy())
            y.extend(rank[ids[keep]].tolist())
    X = np.concatenate(X)[:25000]
    y = np.asarray(y)[:25000]
    sc = StandardScaler().fit(X[:20000])
    clf = LogisticRegression(max_iter=1000)   # multinomial is the default;
    # the explicit multi_class kwarg was removed in scikit-learn 1.7
    clf.fit(sc.transform(X[:20000]), y[:20000])
    scores["wordcontent"] = float(
        clf.score(sc.transform(X[20000:]), y[20000:]))

    try:
        import nltk
        try:
            sents = nltk.corpus.brown.tagged_sents(tagset="universal")
        except LookupError:
            nltk.download("brown"); nltk.download("universal_tagset")
            sents = nltk.corpus.brown.tagged_sents(tagset="universal")
        tags = sorted({tg for s in sents[:4000] for _, tg in s})
        t2i = {tg: i for i, tg in enumerate(tags)}
        Xp, yp = [], []
        with torch.no_grad():
            for s in sents[:4000]:
                words = [w for w, _ in s][:cfg.seq_len]
                e = tk.encode(words, is_pretokenized=True)
                ids = e.ids[:cfg.seq_len]
                if not ids:
                    continue
                wid = e.word_ids[:cfg.seq_len]
                h = enc(torch.as_tensor(
                    np.asarray(ids, dtype=np.int64)[None, :],
                    device=dev))[0].cpu().numpy()
                seen = set()
                for j, w in enumerate(wid):
                    if w is None or w in seen or w >= len(s):
                        continue
                    seen.add(w)          # first subtoken of each word
                    Xp.append(h[j]); yp.append(t2i[s[w][1]])
                if len(yp) >= 25000:
                    break
        Xp = np.asarray(Xp)[:25000]; yp = np.asarray(yp)[:25000]
        sc = StandardScaler().fit(Xp[:20000])
        clf = LogisticRegression(max_iter=1000)
        clf.fit(sc.transform(Xp[:20000]), yp[:20000])
        scores["pos"] = float(clf.score(sc.transform(Xp[20000:]),
                                        yp[20000:]))
    except Exception as exc:            # nltk missing or tokenizer too old
        print("[warn] POS probe skipped:", repr(exc))
        print("[warn] P4 needs it; install nltk and re-run probe.")
        scores["pos"] = None
    return scores


# ---------------------------------------------------------------- driver

def cmd_selftest():
    print("SELFTEST (numpy only)")
    print("  version:", SCRIPT_VERSION)
    rng = np.random.default_rng(0)
    for cell, (k, l) in CELLS.items():
        for _ in range(500):
            pos = sample_mask(rng, 128, k, l)
            d = np.diff(pos)
            blocks = 1 + int((d > 1).sum())
            assert blocks == k, (cell, blocks)
            assert len(pos) == k * l
            runs_len = np.split(pos, np.where(d > 1)[0] + 1)
            assert all(len(r) == l for r in runs_len)
            assert all(np.all(np.diff(r) == 1) for r in runs_len)
    print("  mask sampler: 1500 draws, exact (k, l), no touching -> OK")
    led = run_ledger()
    ids = [r for r, _ in led]
    # count is derived, not hardcoded: an earlier version asserted 34 and
    # failed the moment the P1b curve was appended. The power extension
    # (seeds 5-14 on every 5-seed cell/arm, seeds 3-8 on the two 3-seed
    # objective arms) is a fixed, fully-determined addition, so its size
    # is a worked-out constant rather than re-derived at runtime -- but the
    # duplicate-ID check just below is what actually catches a mistake
    # here, not this arithmetic.
    assert len(ids) == len(set(ids)), "duplicate run id in the ledger"
    n_p1b = len([i for i in ids if i.startswith("P1b_") and
                 int(i.rsplit("seed", 1)[1]) < 5])
    n_power_ext = 92  # 3x(A,B,D) + Aeq + P4mlm + 3x(E,F,G) at 10 new seeds
                       # each (70) + VICReg + LDB at 6 new seeds each (12)
    n_r2 = 15         # pure span-MLM baseline, review round 1
    assert len(ids) == 34 + n_p1b + n_power_ext + n_r2, "unexpected ledger size"
    print("  ledger: %d unique runs (%d pre-registered + %d P1b + %d power "
          "extension + %d review baseline) -> OK"
          % (len(ids), 34, n_p1b, n_power_ext, n_r2))
    # every cell must admit its (k, l) at T = 128: k blocks of length l
    # with at least one gap between them need k*l + (k-1) <= T
    for cell, (k, l) in CELLS.items():
        need = k * l + (k - 1)
        assert need <= Config().seq_len, (
            "cell %s needs %d positions, T is %d"
            % (cell, need, Config().seq_len))
    print("  all %d cells fit at T=%d -> OK" % (len(CELLS), Config().seq_len))
    bad = [c for c in Path(__file__).read_text(encoding="utf-8")
           if ord(c) > 127]
    assert not bad, "non-ascii characters found in this file"
    print("  this file is pure ASCII -> OK")
    p = RUNS / "demo" / "sub"
    print("  path join demo (portable):", str(p))
    print("SELFTEST PASSED")


def cmd_time():
    cfg = replace(Config(), steps=200, data=Config().data)
    print("[time] pre-warming the data cache so the first-run download and")
    print("[time] BPE training are NOT counted in the per-step estimate")
    load_token_stream(cfg)
    t0 = time.time()
    train_run("TIMING_200", cfg)
    per_step = (time.time() - t0) / 200.0
    full = per_step * 9000 / 60.0
    total = 0.0
    for rid, c in run_ledger():
        total += per_step * c.steps / 60.0
    print("MEASURED: %.0f ms/step -> %.1f min per 9k run" %
          (per_step * 1000, full))
    print("RECOMPUTED LEDGER TOTAL: %.1f h (spec padding was 21.0 h)"
          % (total / 60.0))
    (RESULTS / "TIMING_200.json").unlink(missing_ok=True)


def cmd_all():
    print("[version]", SCRIPT_VERSION)
    for rid, cfg in run_ledger():
        done = RESULTS / (rid + ".json")
        if done.exists():
            saved = json.load(open(done, encoding="utf-8"))
            if saved["config"].get("data") == "synthetic":
                sys.exit("results/%s.json is a SYNTHETIC smoke-test "
                         "leftover. Delete it (and runs/%s/) before "
                         "running all, or its probe scores would "
                         "silently pollute the real results." % (rid, rid))
            if "probes" in saved:
                print("[skip]", rid)
                continue
        else:
            train_run(rid, cfg)
        probe_run(rid)


def cmd_aggregate():
    print("[version]", SCRIPT_VERSION)
    if not RESULTS.exists():
        sys.exit("no results directory yet -- nothing has been run")
    rows = []
    for f in sorted(RESULTS.glob("*.json")):
        r = json.load(open(f, encoding="utf-8"))
        pr = r.get("probes", {})
        dg = r["diagnostics"][-1] if r.get("diagnostics") else {}
        rows.append((r["run_id"], pr.get("agnews", ""), pr.get("sst2", ""),
                     pr.get("wordcontent", ""), pr.get("pos", ""),
                     dg.get("lam_min", ""), dg.get("eff_rank", ""),
                     dg.get("w_frob", ""), r.get("minutes", "")))
    out = RESULTS / "summary.csv"
    with open(out, "w", encoding="utf-8") as f:
        f.write("run_id,agnews,sst2,wordcontent,pos,lam_min,eff_rank,w_frob,minutes\n")
        for row in rows:
            f.write(",".join(str(x) for x in row) + "\n")
    print("wrote", str(out), "(%d rows)" % len(rows))


def cmd_clean(wipe_experiment):
    """Safe cleanup. Never touches data/ -- that cache is the most
    expensive thing in the project and is identical across attempts."""
    import shutil
    removed = []
    for p in [RUNS / "TIMING_200", ROOT / "__pycache__"]:
        if p.exists():
            shutil.rmtree(p)
            removed.append(str(p.relative_to(ROOT)))
    for p in RESULTS.glob("SMOKE_*.json") if RESULTS.exists() else []:
        p.unlink()
        removed.append(str(p.relative_to(ROOT)))
    print("removed:", ", ".join(removed) if removed else "nothing to remove")
    if not wipe_experiment:
        print("kept: data/ (expensive cache), runs/ and results/ (resume "
              "state).")
        print("to erase ALL trained runs and start the ledger from zero:")
        print("  python p1_experiment.py clean --wipe-experiment")
        return
    n_done = len(list(RESULTS.glob("*.json"))) if RESULTS.exists() else 0
    print("--wipe-experiment will DELETE runs/ and results/"
          " (%d finished run files)." % n_done)
    print("data/ will be kept. Everything deleted must be retrained"
          " (up to ~7 hours).")
    answer = input("type WIPE to confirm: ")
    if answer.strip() == "WIPE":
        for p in (RUNS, RESULTS):
            if p.exists():
                shutil.rmtree(p)
        print("runs/ and results/ deleted. data/ kept.")
    else:
        print("aborted; nothing deleted.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("command", choices=["selftest", "time", "train",
                                        "probe", "all", "aggregate",
                                        "clean"])
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--data", default=None,
                    help="override data source: wikitext | synthetic")
    ap.add_argument("--wipe-experiment", action="store_true",
                    help="with 'clean': also delete runs/ and results/ "
                         "after typed confirmation (data/ is always kept)")
    args = ap.parse_args()
    if args.command == "clean":
        cmd_clean(args.wipe_experiment); return
    if args.command == "selftest":
        cmd_selftest(); return
    if args.command == "time":
        cmd_time(); return
    if args.command == "aggregate":
        cmd_aggregate(); return
    if args.command == "all":
        cmd_all(); return
    led = dict(run_ledger())
    if args.run_id not in led:
        sys.exit("unknown --run-id; valid ids:\n  " +
                 "\n  ".join(sorted(led)))
    cfg = led[args.run_id]
    if args.data:
        cfg = replace(cfg, data=args.data)
    if args.command == "train":
        train_run(args.run_id, cfg)
    else:
        probe_run(args.run_id)


if __name__ == "__main__":
    main()
