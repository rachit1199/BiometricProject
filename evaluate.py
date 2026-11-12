"""FAR / FRR / EER for face-only, voice-only and fused scores, with bootstrap CIs, operating points and DET curve.

Dataset: DATASET/<person>/sample_01.jpg + sample_01.wav ... (same index = same login attempt)
First --enroll samples enrol; the rest are genuine tests; other people's samples are impostor attempts.
Usage: python evaluate.py DATASET [--enroll 3] [--bootstrap 200] [--plot]
Outputs: scores.csv, roc.csv (fused curve), report.json, det.png (with --plot, needs matplotlib)
"""
import argparse, base64, csv, glob, json, os
import numpy as np
import cv2
import face, voice


def curves(g, im):
    ts = np.linspace(0, 1, 1001)
    return ts, np.array([(im >= t).mean() for t in ts]), np.array([(g < t).mean() for t in ts])


def metrics(g, im):
    """Return (EER, threshold at EER)."""
    ts, far, frr = curves(g, im)
    i = int(np.argmin(np.abs(far - frr)))
    return float((far[i] + frr[i]) / 2), float(ts[i])


def op_point(g, im, target, by="far"):
    """FRR at FAR<=target (by='far') or FAR at FRR<=target. Returns (rate, threshold)."""
    ts, far, frr = curves(g, im)
    ok = np.where((far if by == "far" else frr) <= target)[0]
    if len(ok) == 0:
        return float("nan"), float("nan")
    i = ok[0] if by == "far" else ok[-1]
    return float(frr[i] if by == "far" else far[i]), float(ts[i])


def bootstrap_eer(g, im, n, rng):
    es = [metrics(rng.choice(g, len(g)), rng.choice(im, len(im)))[0] for _ in range(n)]
    return float(np.percentile(es, 2.5)), float(np.percentile(es, 97.5))


def load_person(folder):
    out = []
    for jpg in sorted(glob.glob(os.path.join(folder, "*.jpg"))):
        wav = jpg[:-4] + ".wav"
        if os.path.exists(wav):
            with open(wav, "rb") as f:
                y = voice.decode_wav(base64.b64encode(f.read()).decode())
            out.append((face.extract_face(cv2.imread(jpg)), y))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset"); ap.add_argument("--enroll", type=int, default=3)
    ap.add_argument("--bootstrap", type=int, default=200); ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    people = {p: load_person(os.path.join(a.dataset, p)) for p in sorted(os.listdir(a.dataset)) if os.path.isdir(os.path.join(a.dataset, p))}
    rows = []
    for p, samples in people.items():
        enr = samples[:a.enroll]
        enr_faces = [f for f, _ in enr if f is not None]
        if len(enr_faces) < 2 or len(enr) < 2:
            continue
        vp = voice.build_voiceprint([voice.features(y) for _, y in enr])
        for q, qs in people.items():
            for f, y in (qs[a.enroll:] if q == p else qs):
                if f is None:
                    continue
                rows.append(("genuine" if q == p else "impostor", face.match(enr_faces, [f])[0], voice.match(vp, y)[0]))
    with open("scores.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["label", "face", "voice"]); w.writerows(rows)
    arr = np.array([(r[0] == "genuine", r[1], r[2]) for r in rows], dtype=float)
    g, im = arr[:, 0] == 1, arr[:, 0] == 0
    if g.sum() < 2 or im.sum() < 2:
        raise SystemExit("not enough genuine/impostor attempts, add people or samples")
    rng = np.random.default_rng(0)
    print(f"genuine={g.sum():.0f} impostor={im.sum():.0f}\n")
    print(f"{'system':<20}{'EER':>8}{'95% CI':>18}{'thr':>6}{'FRR@FAR1%':>11}{'FAR@FRR1%':>11}")
    report, systems = {}, {"face only": arr[:, 1], "voice only": arr[:, 2]}
    best = None
    for wf in np.linspace(0, 1, 11):
        s = wf * arr[:, 1] + (1 - wf) * arr[:, 2]
        e, t = metrics(s[g], s[im])
        if best is None or e < best[0]:
            best = (e, t, wf)
    systems[f"fused (w_face={best[2]:.1f})"] = best[2] * arr[:, 1] + (1 - best[2]) * arr[:, 2]
    for name, s in systems.items():
        e, t = metrics(s[g], s[im]); lo, hi = bootstrap_eer(s[g], s[im], a.bootstrap, rng)
        r1, _ = op_point(s[g], s[im], 0.01, "far"); r2, _ = op_point(s[g], s[im], 0.01, "frr")
        print(f"{name:<20}{e:>8.2%}{f'[{lo:.1%}, {hi:.1%}]':>18}{t:>6.2f}{r1:>11.2%}{r2:>11.2%}")
        report[name] = {"eer": e, "ci95": [lo, hi], "threshold": t, "frr_at_far_1pct": r1, "far_at_frr_1pct": r2}
    fs = systems[list(systems)[-1]]
    ts, far, frr = curves(fs[g], fs[im])
    with open("roc.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["threshold", "far", "frr"]); w.writerows(zip(ts, far, frr))
    # three-zone policy check using config thresholds
    import config as cfg
    s = cfg.W_FACE * arr[:, 1] + cfg.W_VOICE * arr[:, 2]
    zone = lambda m: ((m >= cfg.T_HIGH).mean(), ((m >= cfg.T_LOW) & (m < cfg.T_HIGH)).mean(), (m < cfg.T_LOW).mean())
    gz, iz = zone(s[g]), zone(s[im])
    print(f"\nCurrent config.py policy (W={cfg.W_FACE}/{cfg.W_VOICE}, T_HIGH={cfg.T_HIGH}, T_LOW={cfg.T_LOW}):")
    print(f"  genuine : grant {gz[0]:.1%}  otp {gz[1]:.1%}  reject {gz[2]:.1%}")
    print(f"  impostor: grant {iz[0]:.1%} (false accept)  otp {iz[1]:.1%}  reject {iz[2]:.1%}")
    print(f"-> set W_FACE={best[2]:.1f}, W_VOICE={1 - best[2]:.1f}, T_HIGH~{best[1]:.2f} in config.py")
    report["policy"] = {"genuine": gz, "impostor": iz}
    json.dump(report, open("report.json", "w"), indent=2)
    if a.plot:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        for name, sc in systems.items():
            _, fa, fr = curves(sc[g], sc[im]); plt.plot(fa * 100, fr * 100, label=name)
        plt.xlabel("FAR %"); plt.ylabel("FRR %"); plt.xscale("symlog", linthresh=1); plt.yscale("symlog", linthresh=1)
        plt.legend(); plt.grid(alpha=.3); plt.title("DET curve"); plt.savefig("det.png", dpi=150)


if __name__ == "__main__":
    main()
