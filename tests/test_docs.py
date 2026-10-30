import numpy as np

from app import bird
from app.docs import Docs, description_dir, load_chunks

HEADER = "original_column_name,column_name,column_description,data_format,value_description\n"


def _write(bird_dir, table, text, encoding="utf-8"):
    folder = description_dir("bank", bird_dir)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{table}.csv").write_bytes(text.encode(encoding))


def test_one_chunk_per_described_column(tmp_path):
    _write(tmp_path, "district", HEADER + "A11,average salary,,integer,\nA3,,region,text,\"Prague, south Bohemia\"\n,,,,\n",
           "utf-8-sig")
    chunks = load_chunks("bank", tmp_path)
    assert [c.id for c in chunks] == ["doc:district.A11", "doc:district.A3"]
    for c in chunks:
        c.validate()
        assert c.provenance == "docs" and c.terms == []
    assert "Name: average salary." in chunks[0].body
    assert "Values: Prague, south Bohemia" in chunks[1].body and "Name:" not in chunks[1].body


def test_headers_with_spaces_and_non_utf8_files(tmp_path):
    header = " Original_Column_Name , column_name ,column_description,data_format,value_description\n"
    _write(tmp_path, "trans", header + "k_symbol,,characterization: pojistné = insurance,text,\n", "cp1252")
    (chunk,) = load_chunks("bank", tmp_path)
    assert chunk.id == "doc:trans.k_symbol" and "pojistné" in chunk.body


def test_missing_folder_gives_no_chunks(tmp_path):
    assert load_chunks("bank", tmp_path) == []
    assert Docs.load("bank", bird_dir=tmp_path).retrieve("anything") == []


def test_retrieves_the_matching_column(tmp_path):
    _write(tmp_path, "district", HEADER + "A11,,average salary,integer,\nA14,,no. of entrepreneurs per 1000 inhabitants,integer,\n")
    _write(tmp_path, "loan", HEADER + "status,,repayment status,text,\"A: contract finished, no problems\"\n")
    docs = Docs.load("bank", bird_dir=tmp_path)
    assert docs.index.mode == "bm25"
    assert docs.retrieve("Which district has the highest average salary?", k=1)[0].id == "doc:district.A11"
    assert docs.retrieve("loans whose contract finished", k=1)[0].id == "doc:loan.status"


def test_embedder_makes_it_hybrid(tmp_path):
    class Fake:
        def embed_query(self, text):
            return np.ones(3, dtype=np.float32)

        def embed_passages(self, texts):
            return np.ones((len(texts), 3), dtype=np.float32)

    _write(tmp_path, "district", HEADER + "A11,,average salary,integer,\n")
    assert Docs.load("bank", Fake(), tmp_path).index.mode == "hybrid"


def test_description_dir_sits_next_to_the_database(tmp_path):
    assert description_dir("bank", tmp_path) == bird.db_path("bank", tmp_path).parent / "database_description"
