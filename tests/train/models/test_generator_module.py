import ralf.train.models.generator as generator


def test_generator_exports() -> None:
    assert hasattr(generator, "CGLGenerator")
    assert hasattr(generator, "MaskGIT")
