from fastapi.testclient import TestClient

from matera.api.server import app

client = TestClient(app)


def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_process_nonexistent_folder():
    response = client.get(
        "/api/process-stream", params={"folder_path": "non_existent_folder_xyz_123"}
    )
    assert response.status_code == 200
    assert "error" in response.text


def test_process_empty_folder(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    response = client.get("/api/process-stream", params={"folder_path": str(empty_dir)})
    assert response.status_code == 200
    assert "error" in response.text
