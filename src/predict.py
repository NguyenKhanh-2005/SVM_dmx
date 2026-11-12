# =============================================================================
# File: src/predict.py
# Mục đích: Dùng mô hình đã train để dự đoán cảm xúc review mới
#
# Yêu cầu:
#   - Phải chạy main.py trước để tạo file trong models/
#   - Không train lại khi chạy file này
#
# Cách chạy (từ thư mục gốc SVM_dmx/):
#   python src/predict.py
# =============================================================================

import os
import sys
import joblib
import numpy as np

# Đảm bảo console Windows hiển thị đúng tiếng Việt (UTF-8)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Thêm thư mục gốc vào sys.path để import được package src
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.preprocessing import clean_text
from src.svm import predict_svm

MODELS_DIR = os.path.join(BASE_DIR, "models")


def load_model(vectorizer_name, svm_name):
    """
    Tải vectorizer (.joblib) và trọng số SVM (.npz).

    Trả về:
        vectorizer : object dùng để biến văn bản thành vector
        w          : vector trọng số
        b          : bias
    """
    vectorizer_path = os.path.join(MODELS_DIR, vectorizer_name)
    svm_path = os.path.join(MODELS_DIR, svm_name)

    # Kiểm tra file tồn tại trước khi tải
    if not os.path.exists(vectorizer_path):
        print(f"Lỗi: Không tìm thấy {vectorizer_path}")
        print("Hãy chạy 'python main.py' trước để train mô hình.")
        sys.exit(1)

    if not os.path.exists(svm_path):
        print(f"Lỗi: Không tìm thấy {svm_path}")
        print("Hãy chạy 'python main.py' trước để train mô hình.")
        sys.exit(1)

    vectorizer = joblib.load(vectorizer_path)

    # np.load() đọc file .npz; dữ liệu được lấy theo tên đã lưu
    svm_data = np.load(svm_path)
    w = svm_data["w"]
    b = float(svm_data["b"])

    return vectorizer, w, b


def predict_sentiment(text, vectorizer, w, b):
    """
    Dự đoán cảm xúc của một review.

    Các bước:
        1. Tiền xử lý bằng đúng hàm clean_text dùng lúc train.
        2. Chuyển văn bản thành vector bằng vectorizer đã lưu.
        3. Tính score = w·x + b và chọn nhãn.
    """
    # Bước 1: tiền xử lý
    cleaned_text = clean_text(text)

    # Bước 2: transform (KHÔNG fit lại)
    vector = vectorizer.transform([cleaned_text])

    # Bước 3: dự đoán bằng SVM tự cài đặt
    label = int(predict_svm(vector, w, b)[0])

    if label == 1:
        result = "Positive (Tích cực)"
    else:
        result = "Negative (Tiêu cực)"

    return label, result


def main():
    print("=" * 55)
    print("  HỆ THỐNG PHÂN LOẠI CẢM XÚC REVIEW SẢN PHẨM")
    print("  Điện Máy Xanh — BoW / TF-IDF + SVM")
    print("=" * 55)

    print("\nChọn mô hình:")
    print("  1. BoW + SVM")
    print("  2. TF-IDF + SVM")
    print("  3. Dùng cả hai mô hình để so sánh")

    choice = input("Nhập lựa chọn (1, 2 hoặc 3): ").strip()

    if choice == "1":
        vectorizer, w, b = load_model(
            "bow_vectorizer.joblib", "bow_svm.npz"
        )
        print("Đã tải mô hình BoW + SVM.")

    elif choice == "2":
        vectorizer, w, b = load_model(
            "tfidf_vectorizer.joblib", "tfidf_svm.npz"
        )
        print("Đã tải mô hình TF-IDF + SVM.")

    elif choice == "3":
        bow_vectorizer, bow_w, bow_b = load_model(
            "bow_vectorizer.joblib", "bow_svm.npz"
        )
        tfidf_vectorizer, tfidf_w, tfidf_b = load_model(
            "tfidf_vectorizer.joblib", "tfidf_svm.npz"
        )
        print("Đã tải cả hai mô hình.")

    else:
        print("Lựa chọn không hợp lệ.")
        return

    print("\nNhập review cần dự đoán. Gõ 'thoat' để kết thúc.")

    while True:
        print("-" * 55)
        review = input("Nội dung review: ").strip()

        if review.lower() == "thoat":
            print("Tạm biệt!")
            break

        if not review:
            print("Vui lòng nhập nội dung review.")
            continue

        print()

        if choice == "1":
            _, result = predict_sentiment(review, vectorizer, w, b)
            print(f"BoW + SVM    → {result}")

        elif choice == "2":
            _, result = predict_sentiment(review, vectorizer, w, b)
            print(f"TF-IDF + SVM → {result}")

        else:
            _, bow_result = predict_sentiment(
                review, bow_vectorizer, bow_w, bow_b
            )
            _, tfidf_result = predict_sentiment(
                review, tfidf_vectorizer, tfidf_w, tfidf_b
            )
            print(f"BoW + SVM    → {bow_result}")
            print(f"TF-IDF + SVM → {tfidf_result}")


if __name__ == "__main__":
    main()
