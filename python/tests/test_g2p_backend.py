"""The MeCab adapter preserves POS tags and avoids runtime pip installs."""

import sys
from types import SimpleNamespace

from python.build.g2p.ko.g2p import KoreanG2P, _MecabPosAdapter


def test_pos_adapter_preserves_compound_tags_and_punctuation():
    tagger = SimpleNamespace(parse=lambda text: (
        "사과\tNNG,*,F,사과,*,*,*,*\n"
        "입니다\tVCP+EF,*,F,입니다,Inflect,*,*,*\n"
        ".\tSF,*,*,*,*,*,*,*\nEOS\n"
    ))
    assert _MecabPosAdapter(tagger).pos("사과입니다.") == [
        ("사과", "NNG"), ("입니다", "VCP+EF"), (".", "SF"),
    ]


def test_pos_adapter_accepts_empty_input():
    tagger = SimpleNamespace(parse=lambda text: "EOS\n")
    assert _MecabPosAdapter(tagger).pos("") == []


def test_g2p_is_lazy_and_bypasses_legacy_dependency_install(monkeypatch):
    calls = []

    class FakeG2p:
        def __init__(self):
            self.check_mecab()
            self.mecab = self.get_mecab()

        def check_mecab(self):
            raise AssertionError("Legacy runtime installer must not run")

        def get_mecab(self):
            raise AssertionError("Legacy backend loader must not run")

        def __call__(self, text):
            return text

    def make_tagger():
        calls.append("load")
        return SimpleNamespace(parse=lambda text: "EOS\n")

    monkeypatch.setitem(sys.modules, "g2pkk", SimpleNamespace(G2p=FakeG2p))
    monkeypatch.setitem(sys.modules, "mecab_ko", SimpleNamespace(Tagger=make_tagger))
    g2p = KoreanG2P()
    assert calls == []
    assert g2p.to_ipa("") == []
    assert calls == []
    assert g2p.apply_rules(" 사과 ") == "사과"
    assert g2p.to_ipa("사과") == ["s", "a", "k", "w", "a"]
    # The current runtime uses the same mecab-free rules as Unity.
    assert calls == []
    assert g2p.apply_rules_g2pkk(" 사과 ") == "사과"
    assert calls == ["load"]
    assert isinstance(g2p._g2p.mecab, _MecabPosAdapter)
