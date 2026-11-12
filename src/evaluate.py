# =============================================================================
# File: src/evaluate.py
# Mục đích: Tính các chỉ số đánh giá và vẽ biểu đồ cho hai mô hình
#
# Chỉ dùng sklearn.metrics để ĐÁNH GIÁ (không dùng sklearn.svm để huấn luyện).
# =============================================================================

import matplotlib
matplotlib.use("Agg")  # backend không cần cửa sổ đồ họa
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


def evaluate_model(y_true, y_pred, model_name):
    """
    Tính và in Accuracy, Precision, Recall, F1-score.

    Dùng average="macro": tính chỉ số cho từng lớp rồi lấy trung bình
    không có trọng số, giúp đánh giá công bằng giữa Positive và Negative.

    Trả về một dict để dùng cho bảng so sánh.
    """
    print(f"\n{'=' * 55}")
    print(f"  KẾT QUẢ: {model_name}")
    print(f"{'=' * 55}")

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"  Accuracy : {accuracy:.4f}  ({accuracy * 100:.2f}%)")
    print(f"  Precision: {precision:.4f}  (Macro)")
    print(f"  Recall   : {recall:.4f}  (Macro)")
    print(f"  F1-score : {f1:.4f}  (Macro)")

    # Classification Report: xem chi tiết từng lớp
    print("\n  Classification Report:")
    print(classification_report(
        y_true, y_pred,
        target_names=["Negative (-1)", "Positive (1)"],
        zero_division=0,
    ))

    return {
        "Mô hình": model_name,
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1-score": round(f1, 4),
    }


def plot_confusion_matrix(y_true, y_pred, model_name, save_path, cmap="Blues"):
    """
    Vẽ và lưu Confusion Matrix.

    Bốn ô của ma trận:
        TN | FP
        ------     TN: Negative dự đoán đúng
        FN | TP     TP: Positive dự đoán đúng
                    FP: dự đoán Positive nhưng thực tế là Negative
                    FN: dự đoán Negative nhưng thực tế là Positive
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=cmap,
        xticklabels=["Negative (-1)", "Positive (1)"],
        yticklabels=["Negative (-1)", "Positive (1)"],
    )
    plt.title(f"Confusion Matrix — {model_name}")
    plt.ylabel("Nhãn thực tế")
    plt.xlabel("Nhãn dự đoán")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()

    print(f"  → Đã lưu: {save_path}")


def plot_loss_curve(loss_history, model_name, save_path):
    """
    Vẽ biểu đồ Loss theo Epoch để quan sát quá trình huấn luyện SVM.
    Loss giảm dần nghĩa là mô hình đang học tốt hơn.
    """
    epochs = range(1, len(loss_history) + 1)

    plt.figure(figsize=(7, 4))
    plt.plot(epochs, loss_history, marker="o", color="steelblue")
    plt.title(f"Loss theo Epoch — {model_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()

    print(f"  → Đã lưu: {save_path}")


def plot_comparison(results, save_path):
    """
    Vẽ biểu đồ cột so sánh Accuracy và F1-score của hai mô hình.

    Tham số:
        results: list gồm 2 dict, mỗi dict là kết quả của một mô hình.
    """
    metrics = ["Accuracy", "F1-score"]

    bow = results[0]
    tfidf = results[1]

    bow_scores = [bow[m] for m in metrics]
    tfidf_scores = [tfidf[m] for m in metrics]

    x = range(len(metrics))
    width = 0.35

    plt.figure(figsize=(7, 5))
    bars1 = plt.bar(
        [i - width / 2 for i in x], bow_scores,
        width=width, label="BoW + SVM", color="steelblue",
    )
    bars2 = plt.bar(
        [i + width / 2 for i in x], tfidf_scores,
        width=width, label="TF-IDF + SVM", color="darkorange",
    )

    plt.ylabel("Giá trị")
    plt.title("So sánh Accuracy và F1-score")
    plt.xticks(list(x), metrics)
    plt.ylim(0, 1.1)
    plt.legend()

    # Ghi giá trị số trên đầu mỗi cột
    for bar in list(bars1) + list(bars2):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{bar.get_height():.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()

    print(f"  → Đã lưu: {save_path}")
