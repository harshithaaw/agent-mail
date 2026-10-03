import os
import pytest

from scripts.data_loading import build_training_set


SPAM_DIR = "data/spamassassin/spam"
HAM_DIR = "data/spamassassin/easy_ham"


@pytest.mark.skipif(
    not (os.path.isdir(SPAM_DIR) and os.path.isdir(HAM_DIR)),
    reason="SpamAssassin dataset is not included in the repository",
)
def test_build_training_set():
    data = build_training_set()

    assert data
    assert any(label == "spam" for _, label in data)
    assert any(label == "ham" for _, label in data)
