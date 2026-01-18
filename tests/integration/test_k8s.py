"""Tests for K8s manifests."""

import pytest


@pytest.mark.integration
def test_api_deployment():
    """Test API deployment manifest."""
    from pathlib import Path

    manifest_path = Path(__file__).parent.parent.parent / "deployment/k8s/emp-researcher-k8s.yaml"

    assert manifest_path.exists()
    assert "Deployment" in manifest_path.read_text()


@pytest.mark.integration
def test_monitoring_deployment():
    """Test monitoring deployment manifest."""
    from pathlib import Path

    manifest_path = Path(__file__).parent.parent.parent / "deployment/k8s/monitoring-k8s.yaml"

    assert manifest_path.exists()
    assert "Prometheus" in manifest_path.read_text()


@pytest.mark.unit
def test_k8s_yaml_syntax():
    """Test K8s YAML syntax."""
    import yaml

    from pathlib import Path

    manifests_dir = Path(__file__).parent.parent.parent / "deployment/k8s"

    for yaml_file in manifests_dir.glob("*.yaml"):
        with open(yaml_file) as f:
            yaml.safe_load(f)


@pytest.mark.unit
def test_namespace_exists():
    """Test namespace definition."""
    from pathlib import Path

    manifest_path = Path(__file__).parent.parent.parent / "deployment/k8s/emp-researcher-k8s.yaml"

    assert "kind: Namespace" in manifest_path.read_text()


@pytest.mark.unit
def test_deployment_replicas():
    """Test deployment replica count."""
    from pathlib import Path

    manifest_path = Path(__file__).parent.parent.parent / "deployment/k8s/emp-researcher-k8s.yaml"

    content = manifest_path.read_text()
    assert "replicas: 3" in content
