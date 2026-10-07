"""Pin the two missing names that currently disable query understanding and metadata routing.

Both failures are swallowed at runtime, so /search silently runs a much simpler path than the
code suggests. Fixing either one switches on large, corpus-specific behaviour (and, for the
metadata index, a cross-tenant leak: KD-SEC-09). Fix them deliberately, with the golden set.
"""
import pytest


@pytest.mark.known_defect
@pytest.mark.xfail(strict=True, reason="KD-SRCH-11: LOCATION_PEERS is missing from semantic_utils; "
                                        "enhance_query raises ImportError on every query")
def test_enhance_query_runs():
    from src.semantic.query_enhancement import enhance_query

    _, meta = enhance_query("NDA signed in Austin 2023")
    assert meta.get("original_query") == "NDA signed in Austin 2023"


@pytest.mark.known_defect
@pytest.mark.xfail(strict=True, reason="KD-SRCH-04: 'threading' is not imported in semantic_components; "
                                        "MetadataIndex() raises NameError. Fix only together with KD-SEC-09")
def test_metadata_index_can_be_constructed():
    from src.semantic.semantic_components import MetadataIndex

    MetadataIndex()
