import re

# ============================================================
# File: src/preprocessing.py
# Mục đích: Chứa hàm tiền xử lý văn bản tiếng Việt
# ============================================================


def clean_text(text):
    """
    Tiền xử lý một đoạn văn bản tiếng Việt.

    Các bước thực hiện:
    1. Chuyển về chữ thường
    2. Xóa URL
    3. Xóa ký tự đặc biệt (giữ lại chữ cái, số, khoảng trắng)
    4. Chuẩn hóa khoảng trắng thừa

    Lưu ý:
    - Giữ nguyên dấu tiếng Việt
    - Giữ nguyên từ phủ định: không, chẳng, chưa, ...
    """

    # Bước 1: Chuyển về chữ thường
    text = text.lower()

    # Bước 2: Xóa URL (ví dụ: http://..., https://...)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Bước 3: Xóa các ký tự đặc biệt, giữ lại chữ cái (kể cả có dấu), số và khoảng trắng
    # \w trong Python 3 đã hỗ trợ Unicode nên giữ được chữ tiếng Việt có dấu
    text = re.sub(r'[^\w\s]', ' ', text)

    # Bước 4: Chuẩn hóa khoảng trắng (xóa khoảng trắng thừa đầu/cuối và giữa)
    text = re.sub(r'\s+', ' ', text).strip()

    return text

