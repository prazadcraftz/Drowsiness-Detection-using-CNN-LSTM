"""Raw inventory uses explicit classes, sorted identities and safe paths."""

import pytest

from evaluation import CLASS_MAPPING
from evaluation.manifest import read_manifest, validate_manifest, write_manifest


def rows(n=60):
    output = []
    from evaluation.manifest import FIELDS
    for name, label in CLASS_MAPPING.items():
        for i in range(n):
            row = dict.fromkeys(FIELDS, "")
            row.update(source_id=f"source-{label}-{i:03}", class_name=name, class_id=label, relative_path=f"{name}/{i}.avi",
                       file_sha256=f"hash-{label}-{i}", group_id=f"hash-{label}-{i}")
            output.append(row)
    return output


def test_mapping_and_manifest_roundtrip_sorted(tmp_path):
    assert CLASS_MAPPING == {"Normal": 0, "Yawning": 1, "Eye Close": 2}
    path = tmp_path / "manifest.csv"
    source = rows(3)[::-1]
    write_manifest(path, source)
    result = read_manifest(path)
    assert result == sorted(source, key=lambda row: row["source_id"])
    with pytest.raises(FileExistsError):
        write_manifest(path, result)


@pytest.mark.parametrize("field,value", [("class_name", "repo"), ("class_id", 2), ("relative_path", "../secret.avi"), ("relative_path", "Yawning/video.avi")])
def test_invalid_classes_and_paths_rejected(field, value):
    source = rows(3)
    source[0][field] = value
    with pytest.raises(ValueError):
        validate_manifest(source)


def test_duplicate_ids_and_conflicting_content_labels():
    source = rows(3)
    with pytest.raises(ValueError):
        validate_manifest(source + source[:1])
    source[3]["file_sha256"] = source[0]["file_sha256"]
    with pytest.raises(ValueError, match="conflicting"):
        validate_manifest(source)
