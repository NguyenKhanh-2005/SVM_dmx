# Thư mục results

Thư mục này chứa kết quả thực nghiệm sau khi chạy `python main.py`:

```text
results/
├── confusion_matrix_bow.png    # Confusion Matrix của BoW + SVM
├── confusion_matrix_tfidf.png  # Confusion Matrix của TF-IDF + SVM
├── loss_bow.png                # Loss theo epoch của BoW + SVM
├── loss_tfidf.png              # Loss theo epoch của TF-IDF + SVM
├── comparison_chart.png        # Biểu đồ so sánh Accuracy và F1-score
└── comparison_results.csv      # Bảng so sánh hai mô hình
```

## Các chỉ số đánh giá

- **Accuracy**: tỷ lệ dự đoán đúng trên toàn bộ tập test.
- **Precision (Macro)**: trong các mẫu dự đoán là Positive, bao nhiêu
  phần trăm thực sự là Positive (tính trung bình cho cả hai lớp).
- **Recall (Macro)**: trong các mẫu thực sự là Positive, bao nhiêu phần
  trăm được mô hình phát hiện.
- **F1-score (Macro)**: trung bình điều hòa của Precision và Recall.

## Lưu ý

- Toàn bộ số liệu được tính thật từ tập test, không tạo số liệu giả.
- Chạy lại `python main.py` sẽ ghi đè các file kết quả cũ.
