# =============================================================================
# File: src/svm.py
# Mục đích: Tự cài đặt thuật toán Linear SVM bằng NumPy + SciPy
#
# File này KHÔNG dùng sklearn.svm (SVC, LinearSVC). Toàn bộ quá trình
# học trọng số w và bias b đều được viết thủ công.
#
# Các kiến thức được thể hiện:
#   1. Siêu phẳng phân tách (hyperplane): w·x + b = 0
#   2. Vector trọng số w
#   3. Bias b
#   4. Hinge Loss
#   5. Regularization L2
#   6. Stochastic Gradient Descent (SGD)
#   7. Quy tắc cập nhật trọng số
#   8. Hàm dự đoán
# =============================================================================

import numpy as np
from scipy import sparse


def compute_loss(X, y, w, b, C):
    r"""
    Tính giá trị hàm mục tiêu trên toàn bộ tập dữ liệu.

    Công thức (theo lý thuyết Soft-Margin SVM):

        L(w, b) = 0.5 * ||w||²  +  C * Σ max(0, 1 - yᵢ * (w·xᵢ + b))
                  \_________/      \___________________________________/
                  Regularization L2              Hinge Loss

    Trong đó:
        - 0.5 * ||w||² : phạt w quá lớn để mô hình đơn giản, tránh overfit
        - Hinge Loss   : phạt các mẫu bị phân loại sai hoặc nằm trong margin
        - C            : hệ số cân bằng giữa hai phần trên
    """
    # score_i = w·x_i + b
    scores = X.dot(w) + b

    # hinge_i = max(0, 1 - y_i * score_i)
    hinge = np.maximum(0.0, 1.0 - y * scores)

    # (1/2)||w||² + C * tổng hinge loss
    return 0.5 * np.dot(w, w) + C * hinge.sum()


def train_svm(X, y, C=1.0, learning_rate=0.05, epochs=15, random_state=42):
    r"""
    Huấn luyện Linear SVM bằng Stochastic Gradient Descent (SGD).

    -------------------------------------------------------------------------
    HÀM MỤC TIÊU

        L(w, b) = 0.5 * ||w||² + C * Σ max(0, 1 - yᵢ * (w·xᵢ + b))

    Để việc cập nhật ổn định, chia cả hai vế cho (C * n), với n là số mẫu.
    Phép chia này không làm đổi nghiệm tối ưu, chỉ đổi thang đo của loss:

        L'(w, b) = (λ/2) * ||w||² + (1/n) * Σ hinge

    với λ = 1 / (C * n).

    -------------------------------------------------------------------------
    GRADIENT VỚI MỘT MẪU (xᵢ, yᵢ)

        Nếu yᵢ·(w·xᵢ + b) ≥ 1  → mẫu đúng, nằm ngoài margin:
            chỉ còn gradient của regularization.

        Nếu yᵢ·(w·xᵢ + b) < 1  → mẫu vi phạm margin:
            ∂L'/∂w = λ·w - yᵢ·xᵢ
            ∂L'/∂b =      - yᵢ

    -------------------------------------------------------------------------
    QUY TẮC CẬP NHẬT

        w ← w - learning_rate * (λ·w - yᵢ·xᵢ)
          = (1 - learning_rate·λ)·w + learning_rate·yᵢ·xᵢ

        b ← b + learning_rate·yᵢ

    Phần (1 - learning_rate·λ) làm w co lại một chút sau mỗi bước
    (regularization L2), còn phần cộng thêm giúp w học từ mẫu hiện tại.

    -------------------------------------------------------------------------
    Tham số:
        X             : ma trận đặc trưng thưa (scipy sparse), (n_samples, n_features)
        y             : nhãn -1 hoặc +1
        C             : hệ số điều chuẩn
        learning_rate : tốc độ học
        epochs        : số vòng lặp qua toàn bộ tập train
        random_state  : hạt giống ngẫu nhiên

    Trả về:
        w, b, loss_history
    """
    # Nhãn phải là số thực để tính toán
    y = np.asarray(y, dtype=np.float64)

    # Ma trận thưa CSR cho phép lấy nhanh từng hàng X[i]
    if sparse.issparse(X):
        X = X.tocsr()

    n_samples, n_features = X.shape

    # Khởi tạo w = 0, b = 0
    w = np.zeros(n_features, dtype=np.float64)
    b = 0.0

    # λ = 1 / (C * n) — hệ số regularization sau khi chuẩn hóa hàm mục tiêu
    lam = 1.0 / (C * n_samples)

    loss_history = []
    rng = np.random.RandomState(random_state)

    print(f"    Số mẫu: {n_samples} | Số đặc trưng: {n_features}")
    print(f"    C={C}, learning_rate={learning_rate}, epochs={epochs}")

    for epoch in range(epochs):

        # Xáo trộn thứ tự mẫu mỗi epoch
        indices = rng.permutation(n_samples)

        # SGD: duyệt từng mẫu một
        for i in indices:
            # Lấy hàng thứ i của ma trận thưa (một review)
            xi = X[i]

            # score = w·x_i + b
            score = xi.dot(w)[0] + b

            # --- Bước 1: regularization L2 ---
            # w ← (1 - learning_rate * λ) * w
            # Làm w nhỏ lại một chút để tránh trọng số quá lớn.
            w *= (1.0 - learning_rate * lam)

            # --- Bước 2: nếu mẫu vi phạm margin thì cập nhật theo hinge loss ---
            # y_i * score < 1  →  mẫu sai hoặc nằm trong margin
            if y[i] * score < 1.0:
                # w[indices] += learning_rate * y_i * x_i
                #
                # Với ma trận thưa:
                #   xi.indices : vị trí các đặc trưng khác 0
                #   xi.data    : giá trị tại các vị trí đó
                # Cách này chỉ cập nhật các đặc trưng thực sự xuất hiện,
                # không cần chuyển cả hàng sang ma trận dày (dense).
                w[xi.indices] += learning_rate * y[i] * xi.data

                # b ← b + learning_rate * y_i
                b += learning_rate * y[i]

        # Tính loss sau mỗi epoch để vẽ biểu đồ
        loss = compute_loss(X, y, w, b, C)
        loss_history.append(loss)

        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch + 1:3d}/{epochs}  |  Loss = {loss:.2f}")

    return w, b, loss_history


def predict_svm(X, w, b):
    """
    Dự đoán nhãn cho các mẫu mới.

    Công thức:
        score = w·x + b

        score >= 0  →  +1 (Positive)
        score <  0  →  -1 (Negative)

    Siêu phẳng w·x + b = 0 chính là đường phân tách hai lớp.
    """
    scores = X.dot(w) + b

    # np.where(điều_kiện, giá_trị_đúng, giá_trị_sai)
    return np.where(scores >= 0, 1, -1).astype(int)
