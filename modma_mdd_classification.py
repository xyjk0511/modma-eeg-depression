"""
MODMA 数据集抑郁症检测 (纯模拟数据 DEMO 版)
这仅是基于脑电特征提取 (Alpha Asymmetry, PSD) 和 SVM 分类的离线方法学演示。
注意：本脚本使用随机生成信号，结论无法作为真实医疗评估依据。
"""
from __future__ import annotations

import logging
import warnings
import numpy as np
import mne
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report

logger = logging.getLogger(__name__)

SFREQ: float = 250.0
N_SAMPLES: int = 500
N_FFT: int = 256
ALPHA_FMIN: float = 8.0
ALPHA_FMAX: float = 13.0
CV_N_SPLITS: int = 5
TEST_SIZE: float = 0.2


def load_modma_data(num_subjects: int = 10) -> tuple[mne.EpochsArray, np.ndarray]:
    """
    [算法演示专用] 生成模拟临床数据的占位函数。
    为演示跑通预处理与分类流程，模拟生成了服从统计学差异的假脑电数据。
    """
    logger.info("正在生成纯模拟的随机测试 EEG 数据...")

    ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2']
    info = mne.create_info(ch_names=ch_names, sfreq=SFREQ, ch_types='eeg')

    n_trials_per_group = num_subjects * 10

    rng = np.random.default_rng(42)
    data_healthy = rng.standard_normal((n_trials_per_group, len(ch_names), N_SAMPLES))
    data_mdd = rng.standard_normal((n_trials_per_group, len(ch_names), N_SAMPLES))
    data_mdd[:, ch_names.index('F4'), :] *= 1.5  # 显著放大特征通道以确保 SVM 起效

    epochs_data = np.concatenate([data_healthy, data_mdd], axis=0)
    labels = np.array([0] * n_trials_per_group + [1] * n_trials_per_group)

    epochs = mne.EpochsArray(epochs_data, info, verbose=False)
    try:
        epochs.set_montage('standard_1020')
    except ValueError as e:
        logger.warning("Montage 设置跳过: %s", e)

    return epochs, labels


def extract_alpha_asymmetry(epochs: mne.EpochsArray) -> np.ndarray:
    logger.info("正在提取频域特征 (Alpha 频段: %.0f-%.0f Hz)...", ALPHA_FMIN, ALPHA_FMAX)
    spectrum = epochs.compute_psd(
        method='welch', fmin=ALPHA_FMIN, fmax=ALPHA_FMAX, n_fft=N_FFT, verbose=False
    )
    psds = spectrum.get_data()
    alpha_power = np.mean(psds, axis=2)

    ch_names = epochs.ch_names
    if 'F3' in ch_names and 'F4' in ch_names:
        f3_alpha = alpha_power[:, ch_names.index('F3')]
        f4_alpha = alpha_power[:, ch_names.index('F4')]
        alpha_asymmetry = (f4_alpha - f3_alpha) / (f4_alpha + f3_alpha)
        return np.column_stack((alpha_power, alpha_asymmetry))
    return alpha_power


def evaluate_classifier(X: np.ndarray, y: np.ndarray) -> None:
    logger.info("\n[进行分类器训练与验证]")
    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', C=1.0, gamma='scale', probability=True))
    ])

    cv = StratifiedKFold(n_splits=CV_N_SPLITS, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')

    logger.info("\n===== [DEMO 分类评估结果] =====")
    logger.info("5折验证准确率: %s", [f'{s:.2%}' for s in scores])
    logger.info("平均准确率:   %.2f%% (± %.2f%%)", np.mean(scores) * 100, np.std(scores) * 100)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42, stratify=y
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    logger.info("\n分类报告 (0:健康对照, 1:抑郁组):\n%s",
                classification_report(y_test, y_pred,
                                      target_names=['Healthy Control (0)', 'MDD Group (1)']))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger.info("====== [DEMO] 抑郁症闭环触发信号：纯模拟数据离线验证 ======")

    epochs, labels = load_modma_data(num_subjects=10)
    logger.info("-> 成功加载【模拟数据】，共 %d 个试验试次。", len(labels))

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        epochs.filter(l_freq=0.5, h_freq=45.0, verbose=False)
    features = extract_alpha_asymmetry(epochs)
    evaluate_classifier(features, labels)


if __name__ == "__main__":
    main()
