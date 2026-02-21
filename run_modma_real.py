"""
MODMA 数据集抑郁症检测 (真实数据验证版)
直接读取 BIDS 格式的真实 EDF 脑电文件
"""
from __future__ import annotations

import logging
import os
import glob
import warnings
import numpy as np
import pandas as pd
import mne
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit, cross_val_score
from sklearn.metrics import classification_report

logger = logging.getLogger(__name__)

BIDS_ROOT: str = os.environ.get(
    "MODMA_BIDS_ROOT",
    r"d:\eeg\MODMA_EEG_BIDS_format\EEG_LZU_2015_2_resting state",
)
WINDOW_SEC: int = 10
RESAMPLE_SFREQ: float = 125.0
N_FFT: int = 256
ALPHA_FMIN: float = 8.0
ALPHA_FMAX: float = 13.0
TEST_SIZE: float = 0.2


def _load_subject(edf_path: str, crop_duration: float) -> tuple[mne.io.RawEDF, list[np.ndarray]]:
    """加载单个受试者 EDF，返回 (raw, segments)。"""
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    if raw.times[-1] > crop_duration:
        raw.crop(tmin=0, tmax=crop_duration)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        raw.filter(l_freq=0.5, h_freq=45.0, verbose=False)
        raw.resample(RESAMPLE_SFREQ, verbose=False)
    data = raw.get_data()
    window_size = int(WINDOW_SEC * raw.info['sfreq'])
    segments = [data[:, s:s + window_size]
                for s in range(0, data.shape[1] - window_size + 1, window_size)]
    return raw, segments


def load_real_modma_data(
    max_subjects: int | None = None, crop_duration: float = 60
) -> tuple[mne.EpochsArray, np.ndarray, np.ndarray]:
    """读取真实的 MODMA 测试集数据"""
    logger.info("正在加载真实 MODMA 数据集 (BIDS 格式: %s)", BIDS_ROOT)

    participants_file = os.path.join(BIDS_ROOT, "participants.tsv")
    if not os.path.exists(participants_file):
        raise FileNotFoundError(f"找不到 participants.tsv: {participants_file}")

    df = pd.read_csv(participants_file, sep=r'\s+')

    if 'participant_id' in df.columns:
        df['participant_id'] = df['participant_id'].astype(str).str.strip()

    df = df[df['group'].isin(['MDD', 'HC'])]
    label_map = {'MDD': 1, 'HC': 0}

    if max_subjects is not None:
        if isinstance(max_subjects, bool) or not isinstance(max_subjects, int):
            raise TypeError(f"max_subjects 必须是整数，当前类型: {type(max_subjects).__name__}")
        if max_subjects < 2 or max_subjects % 2 != 0:
            raise ValueError(f"max_subjects 必须是大于等于 2 的偶数，当前值: {max_subjects}")

        half = max_subjects // 2
        mdd_pool = df[df['group'] == 'MDD']['participant_id'].tolist()
        hc_pool = df[df['group'] == 'HC']['participant_id'].tolist()
        max_balanced = min(len(mdd_pool), len(hc_pool)) * 2
        if max_subjects > max_balanced:
            raise ValueError(
                f"max_subjects={max_subjects} 超出可平衡加载上限 {max_balanced} "
                f"(MDD={len(mdd_pool)}, HC={len(hc_pool)})"
            )
        subjects = mdd_pool[:half] + hc_pool[:half]
    else:
        subjects = df['participant_id'].tolist()

    all_epochs: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[str] = []

    loaded_count = 0
    last_reported_count = 0
    ch_names_ref: list[str] | None = None
    sfreq_ref: float = RESAMPLE_SFREQ

    for sub_id in subjects:
        safe_id = os.path.basename(sub_id)
        edf_pattern = os.path.join(BIDS_ROOT, safe_id, 'eeg', f'{safe_id}_task-Resting-state_eeg.EDF')
        edf_files = glob.glob(edf_pattern)

        if not edf_files:
            edf_pattern = os.path.join(BIDS_ROOT, safe_id, 'eeg', f'{safe_id}_task-Resting-state_eeg.edf')
            edf_files = glob.glob(edf_pattern)

        if not edf_files:
            logger.warning("找不到 %s 的 EDF 文件，跳过此人。", safe_id)
            continue

        edf_path = edf_files[0]
        group_label = df[df['participant_id'] == sub_id]['group'].values[0]

        try:
            raw, segments = _load_subject(edf_path, crop_duration)
            if ch_names_ref is None:
                ch_names_ref = raw.ch_names
                sfreq_ref = raw.info['sfreq']
            for segment in segments:
                all_epochs.append(segment)
                labels.append(label_map[group_label])
                groups.append(sub_id)
            if segments:
                loaded_count += 1
                if loaded_count % 10 == 0 and loaded_count != last_reported_count:
                    logger.info("已加载 %d 名受试者...", loaded_count)
                    last_reported_count = loaded_count
        except (OSError, ValueError, RuntimeError) as e:
            logger.error("处理 %s 时发生错误: %s", sub_id, e)

    if not all_epochs or ch_names_ref is None:
        raise ValueError("未能成功加载任何有效数据，请检查数据路径、文件格式或 max_subjects 参数。")

    logger.info("成功加载完毕，总计可用独立窗口试次: %d", len(labels))

    info = mne.create_info(ch_names=ch_names_ref, sfreq=sfreq_ref, ch_types='eeg')
    epochs = mne.EpochsArray(np.array(all_epochs), info, verbose=False)

    return epochs, np.array(labels), np.array(groups)


def extract_real_alpha_asymmetry(epochs: mne.EpochsArray) -> np.ndarray:
    """
    提取真实数据集中的 Alpha 波 (8-13 Hz) 不对称特征
    在 EGI 128系统下: E24 近似等于 F3, E124 近似等于 F4
    """
    logger.info("正在计算功率谱密度 (PSD) 和 额叶 Alpha 偏侧化 (FAA)...")

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        spectrum = epochs.compute_psd(
            method='welch', fmin=ALPHA_FMIN, fmax=ALPHA_FMAX, n_fft=N_FFT, verbose=False
        )
        psds = spectrum.get_data()
    alpha_power = np.mean(psds, axis=2)

    ch_names = epochs.ch_names
    f3_name = "'E24'" if "'E24'" in ch_names else 'E24'
    f4_name = "'E124'" if "'E124'" in ch_names else 'E124'

    if f3_name in ch_names and f4_name in ch_names:
        f3_alpha = alpha_power[:, ch_names.index(f3_name)]
        f4_alpha = alpha_power[:, ch_names.index(f4_name)]

        denom = f4_alpha + f3_alpha
        alpha_asymmetry = np.where(denom != 0, (f4_alpha - f3_alpha) / denom, 0.0)
        features = np.column_stack((alpha_power, alpha_asymmetry))
        logger.info("成功提取 FAA 特征！使用了电极 %s(F3映射) 和 %s(F4映射)", f3_name, f4_name)
    else:
        logger.warning("无法在真实数据中定位到额叶映射电极，退化使用全局平均功率")
        features = alpha_power

    return features


def run_evaluation(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> None:
    logger.info("\n[进行分类器训练与验证]")
    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', C=1.0, gamma='scale'))
    ])

    unique_groups, first_idx = np.unique(groups, return_index=True)
    group_labels = y[first_idx]
    unique_classes = np.unique(group_labels)
    if len(unique_classes) < 2:
        raise ValueError("受试者级标签仅有单一类别，无法评估。请检查数据或增大 max_subjects。")

    class_group_counts = {int(cls): int(np.sum(group_labels == cls)) for cls in unique_classes}
    min_groups_per_class = min(class_group_counts.values())
    if min_groups_per_class < 2:
        raise ValueError(
            "至少一个类别的独立受试者少于 2 人，"
            "无法执行稳健评估。请调大 max_subjects。"
        )

    n_splits = min(5, min_groups_per_class)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    try:
        scores = cross_val_score(
            clf, X, y, cv=cv, groups=groups, scoring='accuracy', error_score='raise'
        )
    except ValueError as exc:
        raise ValueError(f"交叉验证失败: {exc}。请增大 max_subjects 或检查分组标签平衡。") from exc

    logger.info("%d折跨受试交叉验证准确率 (消除数据泄漏版):", n_splits)
    for i, s in enumerate(scores):
        logger.info("  第 %d 折: %.2f%%", i + 1, s * 100)
    logger.info("平稳准确率: %.2f%% (± %.2f%%)\n", np.mean(scores) * 100, np.std(scores) * 100)

    n_groups = len(unique_groups)
    test_group_count = max(len(unique_classes), int(np.ceil(n_groups * TEST_SIZE)))
    if test_group_count >= n_groups:
        raise ValueError("可用受试者过少，无法构建独立测试集。请增大 max_subjects。")

    test_size = test_group_count / n_groups
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    try:
        train_group_idx, test_group_idx = next(sss.split(unique_groups, group_labels))
    except ValueError as exc:
        raise ValueError(f"无法构建分层独立测试集: {exc}。请增大 max_subjects。") from exc

    train_mask = np.isin(groups, unique_groups[train_group_idx])
    test_mask = np.isin(groups, unique_groups[test_group_idx])

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        raise ValueError("分层后仍出现单一类别训练/测试集，无法输出最终报告。请增大 max_subjects。")

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    logger.info("独立受试测试集最终报告 (0:HC 健康对照, 1:MDD 抑郁组):\n%s",
                classification_report(y_test, y_pred, zero_division=0))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger.info("========== 沃高医疗 MODMA 实战验证系统 ==========")
    epochs, labels, groups = load_real_modma_data(max_subjects=20, crop_duration=60)

    features = extract_real_alpha_asymmetry(epochs)

    run_evaluation(features, labels, groups)
    logger.info("=================================================")


if __name__ == "__main__":
    main()
