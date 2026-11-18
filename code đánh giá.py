"""
So sánh trên CÙNG tập Test (chia theo thời gian 80/20 của notebook) và CÙNG nhóm người dùng:
    1. CF thuần (Vòng 1, TruncatedSVD)         -> get_top_candidates(...)
    2. Hybrid CF + LightGBM (Vòng 2)           -> rank_candidates(...)
    3. Baseline phổ biến (mốc tham chiếu)

Cách dùng: dán file này vào 1 ô code mới ở CUỐI notebook (sau mục 4.5) rồi chạy ô cuối bên dưới
(phần "ĐOẠN CHẠY TRÊN NOTEBOOK"). Các hàm metric ở đầu file không phụ thuộc biến của notebook.

Khác với ô 4.5 gốc:
  * Hybrid lấy THẲNG movieId từ rank_candidates (ô 4.5 gốc ánh xạ ngược qua clean_title,
    dễ nhầm các phim trùng tên/remake và làm mất thứ tự xếp hạng).
  * Chọn người dùng NGẪU NHIÊN có seed (ô gốc lấy 500 user đầu theo userId).
  * Thêm Precision/Recall/NDCG/HitRate@K (K=5,10,20), Coverage, Baseline phổ biến.
  * Có khoảng tin cậy 95% (bootstrap ghép cặp theo người dùng) cho chênh lệch giữa các mô hình.
"""
import numpy as np
import pandas as pd

KS = (5, 10, 20)


# ---------------------------------------------------------------------------
# 1. Metric cho một người dùng
# ---------------------------------------------------------------------------
def user_metrics(ranked_ids, relevant, ks=KS):
    """ranked_ids: danh sách movieId đã xếp hạng (giảm dần). relevant: set movieId user thích ở Test."""
    out = {}
    n_rel = len(relevant)
    for k in ks:
        top = list(ranked_ids[:k])
        hits = np.array([m in relevant for m in top], dtype=float)
        h = hits.sum()
        dcg = float((hits / np.log2(np.arange(2, len(top) + 2))).sum())
        idcg = float((1.0 / np.log2(np.arange(2, min(n_rel, k) + 2))).sum())
        out[f"P@{k}"] = h / k
        out[f"R@{k}"] = h / n_rel
        out[f"NDCG@{k}"] = dcg / idcg if idcg > 0 else 0.0
        out[f"HitRate@{k}"] = float(h > 0)
    return out


# ---------------------------------------------------------------------------
# 2. Chạy đánh giá cho nhiều mô hình trên cùng danh sách user
# ---------------------------------------------------------------------------
def evaluate(users, test_positive, recommenders, n_catalog, ks=KS, verbose=True):
    """
    users          : list userId dùng chung cho mọi mô hình
    test_positive  : dict userId -> set(movieId) rating >= 4 trong Test
    recommenders   : dict tên -> hàm f(userId, max_k) trả list movieId đã xếp hạng (độ dài >= max_k)
    n_catalog      : số phim trong danh mục (để tính Coverage)
    Trả về: (bảng trung bình, bảng từng user dạng dài)
    """
    maxk = max(ks)
    rows, seen = [], {name: {k: set() for k in ks} for name in recommenders}
    for n, u in enumerate(users, 1):
        rel = test_positive[u]
        for name, fn in recommenders.items():
            ranked = list(fn(u, maxk))
            rows.append({"userId": u, "model": name, **user_metrics(ranked, rel, ks)})
            for k in ks:
                seen[name][k].update(ranked[:k])
        if verbose and n % 100 == 0:
            print(f"  đã đánh giá {n}/{len(users)} người dùng")
    per_user = pd.DataFrame(rows)
    summary = per_user.drop(columns="userId").groupby("model").mean().reindex(list(recommenders))
    for k in ks:
        summary[f"Coverage@{k}"] = [len(seen[m][k]) / n_catalog for m in summary.index]
    summary["n_users"] = len(users)
    return summary, per_user


def paired_bootstrap(per_user, model_a, model_b, metric="P@10", n_boot=2000, seed=0):
    """Chênh lệch trung bình (A - B) và khoảng tin cậy 95%, lấy mẫu lại theo người dùng."""
    a = per_user[per_user.model == model_a].set_index("userId")[metric]
    b = per_user[per_user.model == model_b].set_index("userId")[metric].reindex(a.index)
    d = (a - b).values
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def comparison_table(per_user, pairs, metrics=("P@10", "R@10", "NDCG@10", "HitRate@10")):
    rows = []
    for a, b in pairs:
        for m in metrics:
            diff, lo, hi = paired_bootstrap(per_user, a, b, m)
            rows.append({"so sánh": f"{a} - {b}", "metric": m, "chênh lệch": diff, "CI95% thấp": lo,
                         "CI95% cao": hi, "khác biệt có ý nghĩa": (lo > 0) or (hi < 0)})
    return pd.DataFrame(rows)


def plot_topk(summary, out_png="chart_topk.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    for ax, key in zip(axes, ["P", "R", "NDCG"]):
        for m in summary.index:
            ax.plot(KS, [summary.loc[m, f"{key}@{k}"] for k in KS], marker="o", label=m)
        ax.set_title(f"{key}@K"); ax.set_xticks(KS); ax.grid(alpha=.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend()
    fig.tight_layout(); fig.savefig(out_png, dpi=160); plt.close(fig)


# ---------------------------------------------------------------------------
# 3. ĐOẠN CHẠY TRÊN NOTEBOOK (dán vào ô cuối; cần các biến/hàm của notebook)
# ---------------------------------------------------------------------------
def run_on_notebook(n_users=1000, seed=42):
    g = globals()
    df_test, user2idx = g["df_test"], g["user2idx"]
    user_watched_train, df_movie_stats = g["user_watched_train"], g["df_movie_stats"]
    get_top_candidates, rank_candidates = g["get_top_candidates"], g["rank_candidates"]

    test_positive = df_test[df_test["target"] == 1].groupby("userId")["movieId"].apply(set).to_dict()
    eligible = sorted(u for u, s in test_positive.items() if u in user2idx and len(s) > 0)
    rng = np.random.default_rng(seed)
    users = list(rng.choice(eligible, size=min(n_users, len(eligible)), replace=False))

    pop_order = df_movie_stats.sort_values("movie_rating_count", ascending=False)["movieId"].tolist()

    def cf_fn(u, k):
        return get_top_candidates(u, top_n=k, exclude_watched=True)[0]

    def hybrid_fn(u, k):
        return rank_candidates(u, top_n_candidates=300, top_k=k)["movieId"].tolist()

    def pop_fn(u, k):
        w = user_watched_train.get(u, set())
        return [m for m in pop_order if m not in w][:k]

    recs = {"CF (Vòng 1)": cf_fn, "Hybrid CF+LightGBM (Vòng 2)": hybrid_fn, "Baseline phổ biến": pop_fn}
    summary, per_user = evaluate(users, test_positive, recs, n_catalog=len(g["movie2idx"]))
    cmp_ = comparison_table(per_user, [("Hybrid CF+LightGBM (Vòng 2)", "CF (Vòng 1)")])
    summary.round(4).to_csv("ket_qua_tong_hop.csv", encoding="utf-8-sig")
    per_user.to_csv("ket_qua_tung_user.csv", index=False)
    cmp_.round(4).to_csv("so_sanh_co_y_nghia.csv", index=False, encoding="utf-8-sig")
    plot_topk(summary)
    return summary, cmp_


# ---------------------------------------------------------------------------
# Tự kiểm tra bằng dữ liệu giả (python 1_code_danh_gia.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # a) kiểm tra metric bằng ví dụ tính tay
    m = user_metrics([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], {2, 4, 99}, ks=(5, 10))
    assert abs(m["P@5"] - 0.4) < 1e-9 and abs(m["R@5"] - 2 / 3) < 1e-9
    dcg = 1 / np.log2(3) + 1 / np.log2(5)
    idcg = 1 + 1 / np.log2(3) + 1 / np.log2(4)
    assert abs(m["NDCG@5"] - dcg / idcg) < 1e-9 and m["HitRate@5"] == 1.0
    # b) chạy thử toàn bộ luồng với mô hình giả
    rng = np.random.default_rng(0)
    tp = {u: set(rng.choice(200, 15, replace=False).tolist()) for u in range(100)}
    good = lambda u, k: list(tp[u])[:3] + [x for x in range(200) if x not in tp[u]][: k - 3]
    bad = lambda u, k: [x for x in range(200) if x not in tp[u]][:k]
    s, pu = evaluate(list(range(100)), tp, {"tốt": good, "kém": bad}, n_catalog=200, verbose=False)
    assert s.loc["tốt", "HitRate@10"] == 1.0 and s.loc["kém", "HitRate@10"] == 0.0
    c = comparison_table(pu, [("tốt", "kém")])
    assert c["khác biệt có ý nghĩa"].all()
    print(s.round(3).T.head(8)); print("Tự kiểm tra: OK")
