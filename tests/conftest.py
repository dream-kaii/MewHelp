import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    app.state.chat_model = None
    app.state.extract_model = None
    with TestClient(app) as c:
        yield c
