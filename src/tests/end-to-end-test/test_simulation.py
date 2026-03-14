from archeo.preset.simulation import get_binary_generation_pipeline


SAMPLE_SIZE = 1000


# NOTE: We do not test the precession fits here because of the long runtime.


def test_run_agnostic_aligned_spin_simulation():

    pipeline = get_binary_generation_pipeline("agnostic_aligned_spin")
    df_binaries, binary_generator = pipeline(size=SAMPLE_SIZE, n_workers=1)

    assert len(df_binaries) == SAMPLE_SIZE
    assert binary_generator is not None


def test_run_1g1g_aligned_spin_simulation():

    pipeline = get_binary_generation_pipeline("2g_aligned_spin")
    df_binaries, binary_generator = pipeline(size=SAMPLE_SIZE, n_workers=1)

    assert len(df_binaries) == SAMPLE_SIZE
    assert binary_generator is not None


def test_run_2g1g_aligned_spin_simulation():

    pipeline = get_binary_generation_pipeline("2g_aligned_spin")
    df_1g1g_binaries, _ = pipeline(size=SAMPLE_SIZE, n_workers=1)

    pipeline = get_binary_generation_pipeline("ng_aligned_spin")
    df_2g1g_binaries, binary_generator = pipeline(df_bh1_binaries=df_1g1g_binaries, size=SAMPLE_SIZE, n_workers=1)

    assert len(df_2g1g_binaries) == SAMPLE_SIZE
    assert binary_generator is not None


def test_run_2g2g_aligned_spin_simulation():

    pipeline = get_binary_generation_pipeline("2g_aligned_spin")
    df_1g1g_binaries, _ = pipeline(size=SAMPLE_SIZE, n_workers=1)

    pipeline = get_binary_generation_pipeline("ng_aligned_spin")
    df_2g2g_binaries, binary_generator = pipeline(
        df_bh1_binaries=df_1g1g_binaries, df_bh2_binaries=df_1g1g_binaries, size=SAMPLE_SIZE, n_workers=1
    )

    assert len(df_2g2g_binaries) == SAMPLE_SIZE
    assert binary_generator is not None
