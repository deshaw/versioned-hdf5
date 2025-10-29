def pytest_report_header(config):  # noqa: ARG001
    import h5py
    import ndindex

    return f"project deps: h5py-{h5py.__version__}, ndindex-{ndindex.__version__}"
