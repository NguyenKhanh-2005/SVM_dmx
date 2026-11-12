# Thư mục models

Thư mục này chứa các mô hình sau khi chạy `python main.py`:

```text
models/
├── bow_vectorizer.joblib    # Từ điển BoW đã học từ tập train
├── bow_svm.npz              # Trọng số w, bias b của SVM (BoW)
├── tfidf_vectorizer.joblib  # Từ điển + trọng số IDF của TF-IDF
└── tfidf_svm.npz            # Trọng số w, bias b của SVM (TF-IDF)
```

## Ý nghĩa các file

- **`*_vectorizer.joblib`**: lưu bằng `joblib`. Chứa từ điển (vocabulary)
  và các tham số đã học của vectorizer. Khi dự đoán, phải dùng đúng
  vectorizer của mô hình tương ứng.
- **`*_svm.npz`**: lưu bằng `numpy.savez`. Chứa:
  - `w`: vector trọng số của siêu phẳng.
  - `b`: bias.
  - `C`, `learning_rate`, `epochs`, `batch_size`: tham số huấn luyện.

## Lưu ý

- Các file mô hình được tạo tự động, không cần tạo thủ công.
- Xóa các file trong thư mục này rồi chạy lại `python main.py` nếu muốn
  train lại từ đầu.
- Các file `.joblib` và `.npz` đã được thêm vào `.gitignore` vì kích thước
  lớn và có thể tạo lại bằng cách chạy `main.py`.
