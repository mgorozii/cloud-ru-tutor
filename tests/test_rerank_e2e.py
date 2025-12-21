import os
from pathlib import Path
import tomllib
import unittest

from services.rag import RAG


class RerankE2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        secrets_path = (
            Path(__file__).resolve().parent.parent / ".streamlit" / "secrets.toml"
        )
        if not secrets_path.exists():
            raise unittest.SkipTest("secrets.toml not found")
        with secrets_path.open("rb") as f:
            data = tomllib.load(f)
        api_key = data.get("GEMINI_API_KEY")
        if not api_key:
            raise unittest.SkipTest("GEMINI_API_KEY missing")
        os.environ["GEMINI_API_KEY"] = api_key

    def test_rerank_live(self):
        rag = RAG()
        docs = [
            {
                "id": "1",
                "text": "Object Storage — сервис для хранения неструктурированных данных в облаке Cloud.ru",
                "metadata": {"title": "Object Storage"},
                "similarity": 0.9,
            },
            {
                "id": "2",
                "text": "Managed Kubernetes — платформа для запуска контейнеров в Cloud.ru",
                "metadata": {"title": "Managed K8s"},
                "similarity": 0.8,
            },
            {
                "id": "3",
                "text": "Virtual Machines — классические ВМ с гибкими конфигурациями",
                "metadata": {"title": "VMs"},
                "similarity": 0.7,
            },
        ]

        res = rag.rerank("что такое object storage", docs, top_k=2, require_llm=True)

        self.assertTrue(res, "rerank returned empty list")
        self.assertLessEqual(len(res), 2)
        returned_ids = {d.get("id") for d in res}
        self.assertTrue(returned_ids.issubset({"1", "2", "3"}))


if __name__ == "__main__":
    unittest.main()
