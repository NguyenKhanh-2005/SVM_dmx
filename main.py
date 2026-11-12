# =============================================================================
# File: main.py
# Mục đích: Chạy toàn bộ project
#   1. Đọc và tiền xử lý dữ liệu
#   2. Chia train/test (80/20, stratify)
#   3. Huấn luyện BoW + SVM
#   4. Huấn luyện TF-IDF + SVM
#   5. Đánh giá, vẽ biểu đồ, so sánh và lưu mô hình
#
# Cách chạy (từ thư mục gốc SVM_dmx/):
#   python main.py
# =============================================================================

import os
import sys
import joblib
import numpy as np
import pandas as pd

# Đảm bảo console Windows hiển thị đúng tiếng Việt (UTF-8)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split

from src.preprocessing import clean_text
from src.svm import train_svm, predict_svm
from src.evaluate import (
    evaluate_model,
    plot_confusion_matrix,
    plot_loss_curve,
    plot_comparison,
)

# -----------------------------------------------------------------------------
# Đường dẫn thư mục
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "dataset_labeled.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_and_clean_data():
    """
    Đọc CSV và làm sạch dữ liệu.

    Các bước:
        1. Đọc file CSV bằng pandas.
        2. Loại bỏ dòng thiếu text hoặc label.
        3. Làm sạch văn bản bằng clean_text().
        4. Loại bỏ các dòng sau khi làm sạch trở thành chuỗi rỗng.
    """
    print("[1] Đọc và tiền xử lý dữ liệu...")

    df = pd.read_csv(DATA_PATH)
    print(f"    Số dòng ban đầu: {len(df)}")

    # Chỉ dùng cột text (đầu vào) và label (nhãn)
    df = df.dropna(subset=["text", "label"])

    # Áp dụng cùng một hàm tiền xử lý cho toàn bộ dữ liệu
    df["text_clean"] = df["text"].apply(clean_text)

    # Bỏ các dòng không còn nội dung sau khi làm sạch
    df = df[df["text_clean"].str.strip() != ""]

    print(f"    Số dòng hợp lệ: {len(df)}")
    print("    Phân bố nhãn:")
    print(df["label"].value_counts().to_string())

    return df


def run_bow_experiment(X_train_text, X_test_text, y_train, y_test):
    """
    Thí nghiệm 1: BoW + SVM.

    CountVectorizer(ngram_range=(1, 2)) dùng cả unigram và bigram.
    Vectorizer chỉ được fit trên tập train để tránh data leakage.
    """
    print("\n" + "=" * 55)
    print("  THÍ NGHIỆM 1: BOW + SVM")
    print("=" * 55)

    print("\n[2] Trích xuất đặc trưng Bag of Words...")

    vectorizer = CountVectorizer(ngram_range=(1, 2))

    # fit_transform trên train: học từ điển + đếm tần suất
    X_train = vectorizer.fit_transform(X_train_text)

    # Chỉ transform trên test: dùng đúng từ điển đã học từ train
    X_test = vectorizer.transform(X_test_text)

    print(f"    Số đặc trưng: {X_train.shape[1]}")
    print(f"    Ma trận train: {X_train.shape}")

    print("\n[3] Huấn luyện SVM (tự cài đặt)...")

    # Cùng cấu hình SVM với thí nghiệm TF-IDF để so sánh công bằng
    w, b, loss_history = train_svm(
        X_train, y_train,
        C=1.0,
        learning_rate=0.05,
        epochs=15,
        random_state=42,
    )

    y_pred = predict_svm(X_test, w, b)

    print("\n[4] Đánh giá mô hình...")
    results = evaluate_model(y_test, y_pred, "BoW + SVM")

    plot_confusion_matrix(
        y_test, y_pred,
        model_name="BoW + SVM",
        save_path=os.path.join(RESULTS_DIR, "confusion_matrix_bow.png"),
        cmap="Blues",
    )

    plot_loss_curve(
        loss_history,
        model_name="BoW + SVM",
        save_path=os.path.join(RESULTS_DIR, "loss_bow.png"),
    )

    # -------------------------------------------------------------------------
    # Lưu mô hình
    #   - Vectorizer: joblib (lưu từ điển và trọng số đặc trưng)
    #   - SVM: numpy .npz (lưu w, b và các tham số huấn luyện)
    # -------------------------------------------------------------------------
    print("\n[5] Lưu mô hình BoW...")

    joblib.dump(
        vectorizer,
        os.path.join(MODELS_DIR, "bow_vectorizer.joblib"),
    )

    np.savez(
        os.path.join(MODELS_DIR, "bow_svm.npz"),
        w=w,
        b=b,
        C=1.0,
        learning_rate=0.05,
        epochs=15,
    )

    print("    → models/bow_vectorizer.joblib")
    print("    → models/bow_svm.npz")

    return results


def run_tfidf_experiment(X_train_text, X_test_text, y_train, y_test):
    """
    Thí nghiệm 2: TF-IDF + SVM.

    TfidfVectorizer(ngram_range=(1, 2)) dùng cả unigram và bigram.
    Vectorizer chỉ được fit trên tập train để tránh data leakage.

    TF-IDF = TF × IDF:
        TF  = số lần từ xuất hiện / tổng số từ trong văn bản
        IDF = log(tổng số văn bản / số văn bản chứa từ)
    Từ phổ biến có IDF thấp, từ đặc trưng có IDF cao.
    """
    print("\n" + "=" * 55)
    print("  THÍ NGHIỆM 2: TF-IDF + SVM")
    print("=" * 55)

    print("\n[6] Trích xuất đặc trưng TF-IDF...")

    vectorizer = TfidfVectorizer(ngram_range=(1, 2))

    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    print(f"    Số đặc trưng: {X_train.shape[1]}")
    print(f"    Ma trận train: {X_train.shape}")

    print("\n[7] Huấn luyện SVM (tự cài đặt)...")

    # Giữ nguyên cấu hình SVM như thí nghiệm BoW
    w, b, loss_history = train_svm(
        X_train, y_train,
        C=1.0,
        learning_rate=0.05,
        epochs=15,
        random_state=42,
    )

    y_pred = predict_svm(X_test, w, b)

    print("\n[8] Đánh giá mô hình...")
    results = evaluate_model(y_test, y_pred, "TF-IDF + SVM")

    plot_confusion_matrix(
        y_test, y_pred,
        model_name="TF-IDF + SVM",
        save_path=os.path.join(RESULTS_DIR, "confusion_matrix_tfidf.png"),
        cmap="Oranges",
    )

    plot_loss_curve(
        loss_history,
        model_name="TF-IDF + SVM",
        save_path=os.path.join(RESULTS_DIR, "loss_tfidf.png"),
    )

    print("\n[9] Lưu mô hình TF-IDF...")

    joblib.dump(
        vectorizer,
        os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"),
    )

    np.savez(
        os.path.join(MODELS_DIR, "tfidf_svm.npz"),
        w=w,
        b=b,
        C=1.0,
        learning_rate=0.05,
        epochs=15,
    )

    print("    → models/tfidf_vectorizer.joblib")
    print("    → models/tfidf_svm.npz")

    return results


def compare_models(bow_result, tfidf_result):
    """
    Tạo bảng so sánh, lưu CSV, vẽ biểu đồ và đưa ra nhận xét.
    """
    print("\n" + "=" * 55)
    print("  SO SÁNH HAI MÔ HÌNH")
    print("=" * 55)

    comparison = pd.DataFrame([bow_result, tfidf_result])
    comparison = comparison.set_index("Mô hình")

    print("\n[10] Bảng so sánh:")
    print(comparison.to_string())

    csv_path = os.path.join(RESULTS_DIR, "comparison_results.csv")
    comparison.to_csv(csv_path, encoding="utf-8-sig")
    print(f"\n    → Đã lưu bảng: {csv_path}")

    plot_comparison(
        [bow_result, tfidf_result],
        save_path=os.path.join(RESULTS_DIR, "comparison_chart.png"),
    )

    print("\n[11] Nhận xét:")
    if tfidf_result["F1-score"] > bow_result["F1-score"]:
        difference = tfidf_result["F1-score"] - bow_result["F1-score"]
        print(f"    TF-IDF + SVM tốt hơn BoW + SVM (F1 cao hơn {difference:.4f}).")
        print("    TF-IDF giảm trọng số từ phổ biến và làm nổi bật từ đặc trưng,")
        print("    nên mô hình phân biệt cảm xúc tốt hơn trên tập dữ liệu này.")
    elif bow_result["F1-score"] > tfidf_result["F1-score"]:
        difference = bow_result["F1-score"] - tfidf_result["F1-score"]
        print(f"    BoW + SVM tốt hơn TF-IDF + SVM (F1 cao hơn {difference:.4f}).")
        print("    Với tập dữ liệu này, tần suất từ đơn giản đã đủ mô tả cảm xúc.")
    else:
        print("    Hai mô hình có F1-score bằng nhau.")


def main():
    print("=" * 55)
    print("  PHÂN LOẠI CẢM XÚC REVIEW SẢN PHẨM ĐIỆN MÁY XANH")
    print("  BoW + SVM và TF-IDF + SVM (SVM tự cài đặt)")
    print("=" * 55)

    # Bước 1: Đọc và làm sạch dữ liệu
    df = load_and_clean_data()

    # Bước 2: Chia train/test dùng chung cho cả hai thí nghiệm
    print("\n[2] Chia train/test (80% / 20%)...")

    X = df["text_clean"]
    y = df["label"]

    # stratify=y giữ tỷ lệ nhãn giống nhau ở train và test
    # random_state=42 giúp kết quả lặp lại được
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print(f"    Train: {len(X_train_text)} mẫu")
    print(f"    Test : {len(X_test_text)} mẫu")
    print("    → Hai thí nghiệm dùng CÙNG tập train/test để so sánh công bằng.")

    # Bước 3: Thí nghiệm BoW + SVM
    bow_result = run_bow_experiment(X_train_text, X_test_text, y_train, y_test)

    # Bước 4: Thí nghiệm TF-IDF + SVM
    tfidf_result = run_tfidf_experiment(
        X_train_text, X_test_text, y_train, y_test
    )

    # Bước 5: So sánh và lưu kết quả
    compare_models(bow_result, tfidf_result)

    print("\n" + "=" * 55)
    print("  HOÀN TẤT!")
    print("=" * 55)
    print("  Kết quả trong thư mục results/")
    print("  Mô hình trong thư mục models/")
    print("  Chạy dự đoán: python src/predict.py")


if __name__ == "__main__":
    main()
