# Phân Loại Cảm Xúc Review Sản Phẩm Điện Máy Xanh

Bài tập lớn môn Machine Learning / Xử lý ngôn ngữ tự nhiên.

Dự án xây dựng hệ thống phân loại cảm xúc các bình luận, đánh giá sản phẩm
trên Điện Máy Xanh bằng hai phương pháp trích chọn đặc trưng văn bản:

- **Bag of Words (BoW) + SVM**
- **TF-IDF + SVM**

Thuật toán **Linear SVM được tự cài đặt bằng NumPy**, không sử dụng
`sklearn.svm.SVC` hay `sklearn.svm.LinearSVC` để huấn luyện.

---

## 1. Giới thiệu đề tài

Các trang thương mại điện tử nhận được rất nhiều bình luận, đánh giá sản phẩm
mỗi ngày. Việc đọc và phân loại thủ công toàn bộ các đánh giá này rất tốn
thời gian. Dự án này xây dựng một mô hình Machine Learning đơn giản để tự động
phân loại một review thành **tích cực** hoặc **tiêu cực**.

Dự án cũng so sánh hai cách biểu diễn văn bản phổ biến là **BoW** và **TF-IDF**,
qua đó thấy được ảnh hưởng của cách trích chọn đặc trưng đến chất lượng mô hình.

---

## 2. Mục tiêu bài toán

Cho một đoạn review sản phẩm, dự đoán nhãn cảm xúc:

| Nhãn | Ý nghĩa |
|------|---------|
| `1`  | Positive — Cảm xúc tích cực |
| `-1` | Negative — Cảm xúc tiêu cực |

Đầu vào của mô hình chỉ là nội dung văn bản (`text`). Các cột `id`,
`category`, `confidence` **không** được dùng làm đặc trưng huấn luyện.

---

## 3. Dataset

- **File:** `data/dataset_labeled.csv`
- **Số lượng:** khoảng 6.938 bình luận đã được gán nhãn thủ công
- **Nguồn:** thu thập từ website Điện Máy Xanh

Cấu trúc CSV:

```csv
id,category,text,label,confidence
1,dien-thoai,"Đẹp, mượt. Nhân viên nhiệt tình",1,0.9917
4,dien-thoai,"Sao điện thoại dễ không mà cũng tụt pin nhanh",-1,0.9792
```

| Cột | Ý nghĩa |
|-----|---------|
| `id` | ID bình luận |
| `category` | Danh mục sản phẩm |
| `text` | Nội dung review |
| `label` | Nhãn cảm xúc (-1 hoặc 1) |
| `confidence` | Độ tin cậy khi gán nhãn |

Chỉ sử dụng `text` làm đầu vào và `label` làm nhãn mục tiêu.

---

## 4. Công nghệ sử dụng

| Thư viện | Mục đích |
|----------|---------|
| `numpy` | Tính toán vector, cập nhật trọng số SVM |
| `scipy.sparse` | Xử lý ma trận đặc trưng thưa |
| `pandas` | Đọc và xử lý dữ liệu CSV |
| `scikit-learn` | CountVectorizer, TfidfVectorizer, chia train/test, tính metrics |
| `matplotlib` | Vẽ biểu đồ |
| `seaborn` | Vẽ Confusion Matrix |
| `joblib` | Lưu và tải vectorizer |
| `re` | Tiền xử lý văn bản bằng biểu thức chính quy |

Không sử dụng `sklearn.svm`, TensorFlow, PyTorch hoặc các thư viện Deep Learning.

---

## 5. Cấu trúc project

```text
SVM_dmx/
│
├── data/
│   └── dataset_labeled.csv       # Dataset đã gán nhãn
│
├── src/
│   ├── preprocessing.py          # Hàm làm sạch văn bản
│   ├── svm.py                    # Linear SVM tự cài đặt bằng NumPy
│   ├── evaluate.py               # Đánh giá và vẽ biểu đồ
│   └── predict.py                # Dự đoán review mới
│
├── models/
│   └── README.md                 # Hướng dẫn thư mục models
│
├── results/
│   └── README.md                 # Hướng dẫn thư mục results
│
├── main.py                       # File chạy chính
├── requirements.txt              # Danh sách thư viện
├── README.md                     # File này
└── .gitignore
```

Sau khi chạy `python main.py`, thư mục `models/` và `results/` sẽ có thêm
các file mô hình và kết quả thực nghiệm.

---

## 6. Tiền xử lý văn bản

Hàm `clean_text()` trong `src/preprocessing.py` thực hiện các bước đơn giản:

1. **Chuyển về chữ thường:** `"Điện Thoại ĐẸP"` → `"điện thoại đẹp"`.
2. **Xóa URL:** loại bỏ `http://...`, `https://...`, `www...`.
3. **Xóa ký tự đặc biệt:** giữ lại chữ cái (kể cả có dấu), chữ số và khoảng trắng.
4. **Chuẩn hóa khoảng trắng:** xóa khoảng trắng thừa đầu, cuối và giữa các từ.

Lưu ý:

- Giữ nguyên dấu tiếng Việt.
- Không xóa các từ phủ định như `không`, `chẳng`, `chưa`.
- Không loại bỏ stopwords một cách máy móc.
- Dùng **cùng một hàm** `clean_text()` cho cả lúc train và lúc dự đoán.

---

## 7. Lý thuyết Bag of Words (BoW)

**Bag of Words** biểu diễn văn bản bằng số lần xuất hiện của từng từ.
Thứ tự các từ không được giữ lại — văn bản giống như một "túi từ".

Với cấu hình unigram + bigram:

```python
CountVectorizer(ngram_range=(1, 2))
```

Ví dụ câu `"điện thoại rất đẹp"`:

- **Unigram** (1 từ): `điện`, `thoại`, `rất`, `đẹp`
- **Bigram** (2 từ liên tiếp): `điện thoại`, `thoại rất`, `rất đẹp`

Bigram giúp mô hình nắm được cụm từ có nghĩa như `"pin trâu"`, `"màn hình đẹp"`,
thay vì chỉ học từng từ riêng lẻ.

**Ưu điểm:** đơn giản, dễ hiểu, chạy nhanh.
**Nhược điểm:** không phân biệt từ phổ biến và từ quan trọng; vector thường rất thưa.

---

## 8. Lý thuyết TF-IDF

**TF-IDF** (Term Frequency — Inverse Document Frequency) là cải tiến của BoW.
Mỗi từ được gán một trọng số phản ánh mức độ quan trọng của nó.

Công thức:

```text
TF(t, d)  = số lần từ t xuất hiện trong văn bản d / tổng số từ trong d
IDF(t)    = log(tổng số văn bản / số văn bản có chứa từ t)
TF-IDF    = TF(t, d) × IDF(t)
```

Ý nghĩa:

- Từ xuất hiện nhiều trong một review → TF cao.
- Từ xuất hiện ở ít review → IDF cao (mang nhiều thông tin phân biệt).
- Từ xuất hiện ở hầu hết review (ví dụ `và`, `là`, `có`) → IDF thấp.

Cấu hình:

```python
TfidfVectorizer(ngram_range=(1, 2))
```

**Ưu điểm:** giảm ảnh hưởng của từ phổ biến, làm nổi bật từ đặc trưng.
**Nhược điểm:** phức tạp hơn BoW, vẫn chưa hiểu được ngữ nghĩa và ngữ cảnh sâu.

---

## 9. Lý thuyết SVM

**Support Vector Machine (SVM)** là thuật toán học có giám sát dùng để phân loại.
Với bài toán hai lớp, SVM tìm một **siêu phẳng** (hyperplane) chia dữ liệu
thành hai phần sao cho khoảng cách từ siêu phẳng đến các điểm gần nhất
(margin) là lớn nhất.

Siêu phẳng có phương trình:

```text
f(x) = w·x + b = 0
```

Trong đó:

- `w`: vector trọng số (weight vector).
- `b`: bias (hệ số chặn).
- `w·x`: tích vô hướng giữa vector trọng số và vector đặc trưng.

Quy tắc dự đoán:

```text
Nếu w·x + b >= 0  →  nhãn +1 (Positive)
Nếu w·x + b <  0  →  nhãn -1 (Negative)
```

Với dữ liệu văn bản, số đặc trưng rất lớn (hàng chục nghìn từ và bigram),
nên dùng **Linear SVM** là phù hợp và hiệu quả.

### Vì sao cần Soft-Margin?

Dữ liệu thực tế thường không tách hoàn toàn bằng một đường thẳng. Soft-Margin SVM
cho phép một số mẫu nằm sai phía hoặc trong vùng margin, nhưng bị phạt bởi
Hinge Loss. Nhờ vậy mô hình không bị overfit vào các mẫu nhiễu.

---

## 10. Hinge Loss và quá trình tối ưu

### Hinge Loss

Với mỗi mẫu `(xᵢ, yᵢ)`, `yᵢ ∈ {-1, +1}`:

```text
Lᵢ = max(0, 1 - yᵢ · (w·xᵢ + b))
```

- Nếu `yᵢ · (w·xᵢ + b) ≥ 1`: mẫu được phân loại đúng và nằm ngoài margin
  → `Lᵢ = 0`, không bị phạt.
- Nếu `yᵢ · (w·xᵢ + b) < 1`: mẫu bị sai hoặc nằm trong margin
  → `Lᵢ > 0`, bị phạt.

### Hàm mục tiêu

```text
L(w, b) = 0.5 · ||w||² + C · Σ max(0, 1 - yᵢ · (w·xᵢ + b))
          \___________/     \___________________________________/
           Regularization L2          Hinge Loss
```

- `0.5 · ||w||²`: phạt trọng số quá lớn, giúp mô hình đơn giản, tránh overfit.
- `C`: hệ số cân bằng giữa margin rộng và việc phạt lỗi.
  - `C` lớn → phạt lỗi nhiều, mô hình bám sát dữ liệu train hơn.
  - `C` nhỏ → ưu tiên margin rộng, chấp nhận nhiều lỗi hơn.

### Gradient và quy tắc cập nhật

Vì Hinge Loss không khả vi tại điểm `yᵢ·f(xᵢ) = 1`, ta dùng **sub-gradient**:

```text
∂L/∂w = w - C · Σ yᵢ·xᵢ     (chỉ cộng các mẫu vi phạm margin)
∂L/∂b =   - C · Σ yᵢ        (chỉ cộng các mẫu vi phạm margin)
```

Cập nhật theo Gradient Descent:

```text
w ← w - learning_rate · ∂L/∂w
b ← b - learning_rate · ∂L/∂b
```

Trong code, các phép tính được thực hiện trên từng **mini-batch** để nhanh hơn
và ổn định hơn. Đây là biến thể **Mini-batch Stochastic Gradient Descent (SGD)**.

### Cài đặt

Toàn bộ thuật toán nằm trong `src/svm.py`:

- `compute_loss()`: tính Hinge Loss + regularization.
- `train_svm()`: khởi tạo `w = 0`, `b = 0`, duyệt các epoch, cập nhật trọng số.
- `predict_svm()`: tính `w·x + b` và trả về nhãn `-1` hoặc `+1`.

Ma trận đặc trưng được giữ ở dạng **thưa (scipy.sparse CSR)**, không chuyển
sang ma trận dày, giúp tiết kiệm bộ nhớ khi số đặc trưng lớn.

---

## 11. Các chỉ số đánh giá

| Chỉ số | Ý nghĩa |
|--------|---------|
| **Accuracy** | Tỷ lệ dự đoán đúng trên tổng số mẫu |
| **Precision** | Trong các mẫu dự đoán là Positive, bao nhiêu phần trăm đúng |
| **Recall** | Trong các mẫu thực sự là Positive, bao nhiêu phần trăm được phát hiện |
| **F1-score** | Trung bình điều hòa của Precision và Recall |
| **Confusion Matrix** | Bảng thể hiện số mẫu đúng/sai theo từng nhãn |

Dự án dùng **Macro Precision, Macro Recall, Macro F1** — tính riêng cho từng
lớp rồi lấy trung bình không có trọng số, giúp đánh giá công bằng giữa
Positive và Negative.

Ngoài ra chương trình vẽ **biểu đồ Loss theo Epoch** để quan sát quá trình
huấn luyện của SVM.

Tất cả số liệu đều được tính thật từ tập test, không tạo kết quả giả.

---

## 12. Hướng dẫn cài đặt trên Windows

### Bước 1: Kiểm tra Python

Mở PowerShell và chạy:

```powershell
python --version
```

Khuyến nghị Python 3.10 trở lên.

### Bước 2: Mở project trong VS Code

Mở thư mục `SVM_dmx/` bằng VS Code.

### Bước 3: Tạo môi trường ảo (khuyến khích)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Nếu PowerShell báo lỗi quyền chạy script:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### Bước 4: Cài thư viện

```powershell
pip install -r requirements.txt
```

---

## 13. Hướng dẫn train

Tại thư mục gốc project, chạy:

```powershell
python main.py
```

Chương trình sẽ:

1. Đọc và tiền xử lý `data/dataset_labeled.csv`.
2. Chia train/test 80/20 với `random_state=42`, `stratify=y`.
3. Train **BoW + SVM** và **TF-IDF + SVM** trên cùng tập train/test.
4. In Accuracy, Precision, Recall, F1-score và Classification Report.
5. Vẽ Confusion Matrix và biểu đồ Loss theo Epoch.
6. Vẽ biểu đồ so sánh Accuracy và F1-score.
7. Lưu bảng so sánh và mô hình.

Kết quả nằm trong:

```text
results/
├── confusion_matrix_bow.png
├── confusion_matrix_tfidf.png
├── loss_bow.png
├── loss_tfidf.png
├── comparison_chart.png
└── comparison_results.csv
```

Mô hình nằm trong:

```text
models/
├── bow_vectorizer.joblib
├── bow_svm.npz
├── tfidf_vectorizer.joblib
└── tfidf_svm.npz
```

---

## 14. Hướng dẫn dự đoán

Sau khi đã train xong, chạy:

```powershell
python src/predict.py
```

Chương trình cho phép chọn:

1. BoW + SVM.
2. TF-IDF + SVM.
3. Dùng cả hai mô hình để so sánh.

Ví dụ:

```text
Chọn mô hình:
  1. BoW + SVM
  2. TF-IDF + SVM
  3. Dùng cả hai mô hình để so sánh
Nhập lựa chọn (1, 2 hoặc 3): 3

Nội dung review: Điện thoại đẹp, pin rất trâu, dùng rất mượt.

BoW + SVM    → Positive (Tích cực)
TF-IDF + SVM → Positive (Tích cực)
```

Gõ `thoat` để kết thúc chương trình.

---

## Đẩy project lên GitHub

```powershell
git init
git add .
git commit -m "Hoan thanh project SVM phan loai cam xuc"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/SVM_dmx.git
git push -u origin main
```

Thay `YOUR_USERNAME` bằng tên tài khoản GitHub của bạn.

Các file mô hình và kết quả đã được thêm vào `.gitignore` để tránh đẩy
file nặng lên GitHub. Người khác có thể tạo lại bằng cách chạy `python main.py`.
