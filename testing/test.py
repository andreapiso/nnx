def run(verbose=1):
    import pytest

    pytest_args = ["-l"]

    if verbose and int(verbose) > 1:
        pytest_args += ["-" + "v" * (int(verbose) - 1)]

        pytest_args += ["--pyargs", "nnx"]

        try:
            code = pytest.main(pytest_args)
        except SystemExit as exc:
            code = exc.code

        return code == 0


if __name__ == "__main__":
    run()


