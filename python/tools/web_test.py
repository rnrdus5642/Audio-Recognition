"""Unity의 청취 흐름을 테스트하는 실시간 마이크 전용 Gradio UI.

정답 입력 → 모델 준비 → 마이크 청취 → 연속 N회 정답 확정 → 자동 종료.
현재 웹은 Python 인식·매칭 경로를 사용하며 공통 C# 런타임 연결 전이다.

사용법:
    python -m python.tools.web_test
    python -m python.tools.web_test --port 7860 --share

모델은 준비 버튼 또는 첫 인식 때 로드한다. 최초 사용 시 HuggingFace
모델(~1.2GB)을 다운로드하며, 이후에는 캐시를 재사용한다.
"""

from __future__ import annotations

import argparse
import sys
from html import escape
from pathlib import Path

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

from python.runtime.audio import HopFramer
from python.tools.test_real_audio import AudioTester
from python.tools._web_feedback import pronunciation_feedback_html
from python.tools._web_syllable_feedback import confirmed_feedback_html, model_errors_for


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_RATE = 16_000
LIVE_HOP_S = 0.5  # how often the UI hands audio over

# What makes a frame. The browser decides how big a chunk is and how
# often one arrives; neither may decide how often a child is scored, so
# frames are cut on this many new samples instead.
LIVE_HOP_SAMPLES = int(LIVE_HOP_S * SAMPLE_RATE)



def _normalize_audio_input(audio_in: tuple[int, np.ndarray] | None) -> np.ndarray:
    """Convert Gradio's (sample_rate, ndarray) into the float32 16k mono
    array our pipeline expects.
    """
    if audio_in is None:
        return np.zeros(0, dtype=np.float32)
    sr, audio = audio_in
    audio = np.asarray(audio)
    if audio.dtype.kind == "i":  # int -> float
        audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
    elif audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    return audio.astype(np.float32)


def _confirmed_status_html(hit, fire_time: float) -> str:
    """Show the frame that confirmed the answer and stopped listening."""
    how = (
        "정확일치 1프레임으로 확정 (연속 2회를 기다리지 않음)"
        if hit.streak < getattr(hit, "required_frames", 2)
        else f"연속 {hit.streak}회 통과로 확정"
    )
    return (
        "<div style='padding:16px;border-radius:8px;background:#16a34a;"
        "color:white;'>"
        "<h2 style='margin:0;'>✓ 확정</h2>"
        f"<p style='margin:4px 0;font-size:18px;'>"
        f"<b>{escape(hit.result.target_text or '')}</b> — {fire_time:.1f}초 지점 "
        f"({hit.frames}번째 프레임, 점수 {hit.result.score:.3f})</p>"
        "<p style='margin:4px 0;font-size:14px;opacity:.9;'>"
        f"{how}. 청취를 종료했습니다.</p></div>"
    )



def _score_plot_df(points: list[dict]):
    """Frame for the score-over-time LinePlot.

    `points` is [{t, score, threshold}, ...] in time order; only the
    score is plotted. The thresholds are listed above the chart instead
    of drawn as a second line - they differ per answer word and the
    leading word changes frame to frame, so a threshold line jumps
    around and is harder to read than a fixed reference. (Multi-series
    colour encoding also got silently dropped on streaming updates, so
    that line never rendered in the live tab.)
    """
    import pandas as pd

    rows = [{"시각": p["t"], "점수": p["score"]} for p in points]
    if not rows:
        rows = [{"시각": 0.0, "점수": 0.0}]
    return pd.DataFrame(rows)


def _passing_runs(points: list[dict]) -> list[list[dict]]:
    """Maximal runs of adjacent frames won by the same answer.

    This is exactly what `StreamingMatcher` counts, so the longest run
    is the streak the session actually reached.
    """
    runs: list[list[dict]] = []
    for p in points:
        passed = p["score"] >= p["threshold"] and p.get("word")
        if passed and runs and runs[-1][-1].get("word") == p["word"] \
                and runs[-1][-1]["_i"] == p["_i"] - 1:
            runs[-1].append(p)
        elif passed:
            runs.append([p])
    return runs


def _streak_summary_md(points: list[dict], consecutive: int,
                       confirmed_on_exact: bool = False) -> str:
    """Why the run did (or did not) confirm.

    The chart shows the score crossing its threshold, but confirmation
    also needs `consecutive` frames in a row won by the SAME answer.
    Listing the above-threshold runs makes the difference visible -
    otherwise a chart full of peaks looks like it should have passed.

    One calibrated exception: a single frame scoring exactly 1.0 confirms
    on its own for words the build switched it on for. Saying "needs 2 in
    a row" after that happened would describe a rule the session did not
    use.
    """
    if not points:
        return ""

    runs = _passing_runs(points)
    n_over = sum(1 for p in points if p.get("passed", p["score"] >= p["threshold"]))
    if confirmed_on_exact:
        return (
            f"**정확일치 1프레임으로 확정** — 전체 {len(points)}프레임 중 "
            f"{n_over}개가 임계값을 넘었고, 그중 하나가 점수 1.000 "
            "(정답 음소열과 완전히 같음)이라 연속 2회를 기다리지 않고 "
            "확정했습니다. 이 단어에 보정으로 허용된 예외입니다."
        )
    if not runs:
        return (
            f"통과 조건을 만족한 프레임이 없습니다 "
            f"(전체 {len(points)}프레임)."
        )

    longest = max(len(r) for r in runs)
    word_policy = any("required_frames" in p for p in points)
    requirement = ("단어별 설정 적용" if word_policy
                   else f"확정에 {consecutive}회 필요")
    lines = [
        f"**통과 구간** — 전체 {len(points)}프레임 중 {n_over}개 통과, "
        f"최장 연속 **{longest}회** ({requirement})"
    ]
    for r in runs:
        span = (f"{r[0]['t']:.1f}초" if len(r) == 1
                else f"{r[0]['t']:.1f}~{r[-1]['t']:.1f}초")
        required = r[-1].get("required_frames", consecutive)
        mark = " ✓" if len(r) >= required else ""
        lines.append(
            f"- {span} · **{r[0]['word']}** · {len(r)}프레임"
            f" (점수 {min(p['score'] for p in r):.2f}"
            f"~{max(p['score'] for p in r):.2f}, 필요 {required}회){mark}"
        )
    if not any(len(r) >= r[-1].get("required_frames", consecutive) for r in runs):
        lines.append(
            f"\n➜ 넘기는 했지만 **연속으로 이어지지 않아** 확정되지 "
            "않았습니다. 다시 말한 뒤 판정을 확인하세요."
        )
    return "\n".join(lines)


def _thresholds_md(candidates: list[dict]) -> str:
    """One-line reference of each answer's pass mark.

    Shows the phoneme count, which explains why the numbers differ, and
    where each one came from. A measured threshold and the phoneme-count
    fallback are not near each other - 사과 is 0.875 measured and 0.65 by
    count - and which one is in play decides whether the word confirms on
    speech that never contained it. The number alone does not say, and a
    silent fallback is how this went unnoticed.
    """
    if not candidates:
        return ""
    parts = []
    for c in candidates:
        note = f"{len(c['phonemes'])}음소"
        source = c.get("threshold_source")
        if source == "measured":
            note += " · 측정"
        elif source == "default":
            note += " · 기본값"
        elif source == "configured":
            note += " · 출하 설정"
        # Strictly True, like the matcher: a candidate carrying the string
        # "false" would otherwise be advertised as having the exception
        # while the runtime refuses it.
        if c.get("confirmation_frames") is not None:
            note += f" · 연속 {c['confirmation_frames']}회"
        elif c.get("allow_exact_single_frame") is True:
            note += " · 정확일치 1프레임 허용"
        if c.get("min_whole_score", 0) > 0:
            note += f" · 전체 발음 ≥ {c['min_whole_score']:.2f}"
        # Three decimals: the candidates sit on a 0.025 grid, so two
        # round 0.875 to 0.88 and there is no reading it back against
        # shared/thresholds_child.json.
        parts.append(
            f"**{c['text']}** `{c['threshold']:.3f}` <sub>({note})</sub>"
        )
    return "임계값 &nbsp; " + " &nbsp;·&nbsp; ".join(parts)


def _ipa_md(scored: list[str], dropped: int, window: tuple[int, int] | None,
            target_text: str | None) -> str:
    """Show the phonemes that were actually scored, window marked.

    Only the trailing slice inside the context limit is scored, so
    printing the whole recognised sequence would misrepresent what the
    decision was based on.
    """
    if not scored:
        return "**채점 대상 IPA:** _(없음)_"
    marked = []
    ws, we = window if window else (-1, -1)
    for i, ph in enumerate(scored):
        marked.append(f"**[{ph}]**" if ws <= i < we else f"{ph}")
    out = "**채점 대상 IPA** "
    if dropped:
        out += f"_(문맥 제한으로 앞 {dropped}개 제외)_"
    out += "\n\n" + " ".join(marked)
    if window and target_text:
        out += (
            f"\n\n_굵게 표시된 구간이 **{target_text}**(으)로 매칭된 "
            "윈도우입니다._"
        )
    return out


# Display names for known languages (extend as new languages are added).
LANGUAGE_DISPLAY_NAMES = {
    "ko": "🇰🇷 한국어",
    "en": "🇺🇸 English",
    "ja": "🇯🇵 日本語",
    "zh": "🇨🇳 中文",
    "es": "🇪🇸 Español",
    "fr": "🇫🇷 Français",
    "de": "🇩🇪 Deutsch",
}


def _competitor_md(best) -> str:
    """Why a score over the pass mark was not accepted.

    Without this the panel shows a number above the threshold and a
    verdict of "not yet", which reads as a bug. The child said something;
    it just fits a different action better, and saying which one is the
    only honest way to show a refusal we chose to make.
    """
    if getattr(best, "whole_rejected", False):
        return (
            "<p style='margin:4px 0;font-size:14px;color:#fca5a5;'>"
            "일부 소리는 비슷하지만 전체 발음의 일치도가 부족합니다 "
            f"({best.accepted_whole_score:.2f}). 다시 말해보세요.</p>"
        )
    if not getattr(best, "competitor_rejected", False):
        return ""
    return (
        "<p style='margin:4px 0;font-size:14px;color:#fca5a5;'>"
        f"점수는 넘었지만 <b>{escape(best.competitor_text or '')}</b> 쪽이 "
        "말한 내용 전체에 더 맞아서 인정하지 않았습니다 "
        f"(전체 비교 {best.competitor_score:.2f} vs "
        f"{best.accepted_whole_score:.2f}).</p>"
    )


def _language_label(code: str) -> str:
    return LANGUAGE_DISPLAY_NAMES.get(code, code.upper())


def build_app(tester: AudioTester):
    import gradio as gr

    from python.runtime.recognizer import available_languages as available_recognizers

    usable_languages = sorted(available_recognizers()) or ["ko"]
    language_choices = [
        (_language_label(code), code) for code in usable_languages
    ]
    music_stages = tester.targets.get("presentation_stages", [])

    def _live_candidates(answers_text, lang):
        words = [
            w.strip()
            for w in (answers_text or "").replace(",", "\n").replace("/", "\n").splitlines()
            if w.strip() and not w.strip().startswith("#")
        ]
        cands = []
        for i, w in enumerate(words):
            cand = dict(tester._make_custom_target(w, lang)[1])
            cand["id"] = f"live_{i}"   # unique - the streak keys on id
            cands.append(cand)
        return words, cands

    def live_warmup(answers_text, language):
        """Load the model before streaming starts.

        The first `recognize` call pays for a ~1.2 GB model load. During a
        live stream that stalls the queue and every chunk arrives late, so
        we force it up front.
        """
        lang = language or "ko"
        try:
            words, cands = _live_candidates(answers_text, lang)
            tester._recognizer_for(lang).recognize(
                np.zeros(SAMPLE_RATE, dtype=np.float32)
            )
        except Exception as e:
            return (
                f"<div style='padding:8px;color:#dc2626;'>예열 실패: {e}</div>",
                "",
            )
        return (
            "<div style='padding:8px;background:#dcfce7;border-radius:6px;"
            f"color:#166534;'>준비 완료 — 후보 {', '.join(words) or '(없음)'}. "
            "마이크 버튼을 눌러 말하세요.</div>",
            _thresholds_md(cands),
        )

    def live_reset():
        return (
            {},
            "<div style='padding:8px;color:#6b7280;'>초기화됨. 마이크 버튼을 "
            "눌러 다시 시작하세요.</div>",
            "",
            _score_plot_df([]),
            "",
            _ipa_md([], 0, None, None),
            [],
            pronunciation_feedback_html(None),
            None,
        )

    def live_step(new_chunk, state, answers_text, language,
                  window_s, consecutive):
        """One microphone chunk. Accumulates, re-recognises, confirms."""
        from python.runtime.matching import Matcher, StreamingMatcher

        def unchanged(st, status=None):
            return (
                st,
                status if status is not None else st.get("status", ""),
                _thresholds_md(st.get("cands", [])),
                _score_plot_df(st.get("points", [])),
                _streak_summary_md(st.get("points", []), int(consecutive),
                                   confirmed_on_exact=st.get("confirmed_on_exact", False)),
                st.get("ipa_md", _ipa_md([], 0, None, None)),
                st.get("rows", []),
                st.get("feedback_html", pronunciation_feedback_html(None)),
                st.get("confirmation"),
                gr.update(),
            )

        def failed(reason):
            """Throw this recording away, visibly.

            Half a session is worse than none: the streak in front of the
            failure cannot honestly be joined to whatever passes after
            it, and the numbers on screen would say "1/2" for a frame the
            model never scored. So the audio, the streak and the log all
            go, `done` keeps later chunks away from the model, and the
            child is asked to say it again.
            """
            state["done"] = True
            state["framer"].reset()
            state["sm"] = StreamingMatcher(
                state["matcher"], state["cands"],
                consecutive=int(consecutive),
                feedback_errors=model_errors_for(state["matcher"].matrix.matrix_id))
            state["rows"] = []
            state["points"] = []
            state["elapsed"] = 0.0
            state["ipa_md"] = _ipa_md([], 0, None, None)
            state["confirmation"] = None
            state["feedback_html"] = pronunciation_feedback_html(None)
            state["status"] = (
                "<div style='padding:16px;border-radius:8px;"
                "background:#7f1d1d;color:#fee2e2;'>"
                "<h2 style='margin:0;'>⚠ 채점 실패</h2>"
                f"<p style='margin:4px 0;'>{escape(str(reason))}</p>"
                "<p style='margin:4px 0;font-size:14px;'>이 녹음은 버렸습니다. "
                "마이크 버튼을 다시 눌러 처음부터 말해 주세요.</p></div>"
            )
            return (
                state, state["status"], _thresholds_md(state["cands"]),
                _score_plot_df([]), "", state["ipa_md"], [],
                state["feedback_html"], None, gr.update(recording=False),
            )

        state = state or {}
        # Already confirmed, or nothing to process - leave the view as is.
        if state.get("done") or new_chunk is None:
            return unchanged(state)

        lang = language or "ko"
        keep = int(float(window_s) * SAMPLE_RATE)
        hop = LIVE_HOP_SAMPLES
        key = f"{lang}|{answers_text}|{consecutive}|{keep}|{hop}"
        if state.get("key") != key:
            try:
                words, cands = _live_candidates(answers_text, lang)
            except Exception as e:
                return unchanged(state, (
                    f"<div style='padding:8px;color:#dc2626;'>G2P 실패: {e}"
                    "</div>"))
            if not cands:
                return unchanged(state, (
                    "<div style='padding:8px;color:#dc2626;'>정답 단어를 "
                    "입력하세요.</div>"))
            matcher = Matcher.for_streaming(tester._matrix_for(lang))
            state = {
                "key": key,
                "matcher": matcher,
                "sm": StreamingMatcher(
                    matcher, cands, consecutive=int(consecutive),
                    feedback_errors=model_errors_for(matcher.matrix.matrix_id)
                ),
                "cands": cands,
                # Cuts frames on sample boundaries, not on browser
                # chunks. Matches Unity AudioWindowBuffer: every model
                # input has the full window length, with silence on the
                # left until enough real audio has arrived. Padding does
                # not count toward elapsed time or confirmation frames.
                "framer": HopFramer(keep, hop),
                "rows": [],
                "points": [],
                "elapsed": 0.0,
            }

        chunk = _normalize_audio_input(new_chunk)
        if len(chunk) == 0:
            return unchanged(state)

        frames = state["framer"].feed(chunk)
        if not frames:
            # Part of a hop is not a frame. Keep the samples, score
            # nothing, and leave the display alone: how the browser
            # happens to buffer must not decide how many times a child
            # is judged.
            return unchanged(state)

        try:
            rec = tester._recognizer_for(lang)
            sm = state["sm"]
            hit = None
            for frame in frames:
                _hangul, ipa = rec.recognize_with_text(frame.samples)
                from python.runtime.matching.acoustic import collect_acoustic_evidence
                evidence = collect_acoustic_evidence(rec, state["cands"])
                hit = sm.push(ipa, evidence)
                # The confirmed banner, comparison, and JSON share the hit.
                best = (hit.result if hit
                        else sm.matcher.best_match(ipa, state["cands"], evidence))
                threshold = best.threshold
                # Where the window ends in the audio, not how long the
                # callback has been running: seconds spent waiting for
                # the next chunk are not frames.
                state["elapsed"] = frame.end_seconds(SAMPLE_RATE)
                t = round(state["elapsed"], 1)
                state["rows"] = ([[
                    t, sm.streak, best.target_text or "—",
                    round(best.score, 3), threshold,
                ]] + state["rows"])[:20]
                state["points"].append({
                    "_i": len(state["points"]),
                    "t": t, "score": round(best.score, 3),
                    "threshold": threshold,
                    "word": best.target_text if best.passed else None,
                    "passed": best.passed,
                    **({"required_frames": sm.required_frames(best.target_id)}
                       if any(c.get("confirmation_frames") is not None
                              for c in state["cands"]) else {}),
                })
                scored, dropped = sm.matcher.context_slice(
                    ipa, best.target_phonemes or []
                )
                state["ipa_md"] = _ipa_md(
                    scored, dropped,
                    (best.window_start - dropped, best.window_end - dropped),
                    best.target_text,
                )
                if hit:
                    # The answer is in. Whatever else arrived in this
                    # chunk came after it and is not part of the question.
                    break
        except Exception as e:
            # The frames are spent - `feed` consumed them before anything
            # was scored - and this state object is the one Gradio hands
            # back next time, streak and all. Left alone, a hop that
            # passed before the failure would join a hop that passes
            # after it and confirm on a window nobody ever scored.
            return failed(e)

        plot = _score_plot_df(state["points"])

        thresholds = _thresholds_md(state["cands"])
        runs = _streak_summary_md(
            state["points"], int(consecutive),
            confirmed_on_exact=bool(hit) and hit.streak < sm.required_frames(hit.result.target_id))

        if hit:
            state["done"] = True
            state["confirmed_on_exact"] = hit.streak < hit.required_frames
            state["status"] = _confirmed_status_html(hit, state["elapsed"])
            state["confirmation"] = hit.feedback.to_dict()
            candidate = next(c for c in state["cands"] if c["id"] == hit.result.target_id)
            state["feedback_html"] = confirmed_feedback_html(
                tester, hit.feedback, candidate, lang
            )
            # Stop the mic - this is the "answer accepted, move on" moment.
            return (state, state["status"], thresholds, plot, runs,
                    state["ipa_md"], state["rows"],
                    state["feedback_html"], state["confirmation"],
                    gr.update(recording=False))

        state["status"] = (
            "<div style='padding:16px;border-radius:8px;background:#1f2937;"
            "color:#f3f4f6;'><h2 style='margin:0;'>🎧 듣는 중…</h2>"
            f"<p style='margin:4px 0;font-size:16px;'>{state['elapsed']:.1f}초 · "
            f"연속 <b>{sm.streak}</b>/{sm.required_frames(best.target_id)} · "
            f"선택 {best.target_text or '—'} "
            f"{best.score:.3f} / {threshold:.2f}</p>"
            f"{_competitor_md(best)}</div>"
        )
        return (state, state["status"], thresholds, plot, runs,
                state["ipa_md"], state["rows"],
                pronunciation_feedback_html(None), None, gr.update())

    def select_music_stage(stage_id):
        stage = next(s for s in music_stages if s["id"] == stage_id)
        return ("\n".join(stage["answers"]), *live_reset(), gr.update(recording=False))

    def replay_file(audio, answers_text, language, window_s, consecutive):
        """Replay a file through the live callback at the same 0.5 s hops."""
        pcm = _normalize_audio_input(audio) if audio is not None else np.array([])
        state = {}
        if not len(pcm):
            raise gr.Error("테스트할 음성 파일을 선택하세요.")
        # One second after the recording ends, plus alignment to a hop.
        # This is visible in the upload control; it never counts as speech.
        pcm = np.pad(pcm, (0, (-len(pcm)) % LIVE_HOP_SAMPLES + SAMPLE_RATE))
        for offset in range(0, len(pcm), LIVE_HOP_SAMPLES):
            result = live_step(
                (SAMPLE_RATE, pcm[offset:offset + LIVE_HOP_SAMPLES]), state,
                answers_text, language, window_s, consecutive,
            )
            state = result[0]
            if offset + LIVE_HOP_SAMPLES >= len(pcm) and not state.get("done"):
                result = list(result)
                result[1] = "<div style='padding:16px;'>음성 재생 완료 · 정답 미확정</div>"
                state["status"] = result[1]
                result = tuple(result)
            # This generator updates results, not a streaming Audio output.
            # Returning microphone updates from a file replay makes Gradio
            # try to finish a nonexistent output audio stream.
            yield result[:-1]
            if state.get("done"):
                break

    with gr.Blocks(title="악기놀이 발표 테스트" if music_stages else "실시간 발음 테스트") as app:
        gr.Markdown(
            "# 🔴 실시간 발음 테스트\n"
            "정답 입력 → 준비 → 마이크 청취 → 연속 정답 확정 → 자동 종료"
        )
        gr.Markdown(
            "Unity와 같은 청취 흐름을 테스트합니다. "
            "현재 인식·판정은 Python 경로이며 공통 C# 모듈 연결 전입니다."
        )
        gr.Markdown(
            "말하는 도중 실시간으로 채점합니다. 정답이 **연속 N회** "
            "이기면 **녹음이 자동으로 멈춥니다**.\n\n"
            "⚠ 먼저 **준비** 버튼으로 모델을 예열하세요. 예열 없이 "
            "시작하면 첫 인식에서 모델 로딩을 기다려야 합니다.\n\n"
            f"마이크 입력은 {LIVE_HOP_S}초 간격으로 전달됩니다 "
            "(실제 처리 속도는 실행 환경에 따라 다릅니다)."
        )

        live_state = gr.State({})

        with gr.Row():
            with gr.Column(scale=1):
                lv_language_radio = gr.Radio(
                    choices=language_choices,
                    value=usable_languages[0],
                    label="언어",
                    interactive=True,
                    visible=len(usable_languages) > 1,
                )
                if music_stages:
                    lv_stage = gr.Dropdown(
                        choices=[(f"{s['id']}. {s['title']}", s["id"]) for s in music_stages],
                        value=music_stages[0]["id"], label="악기놀이 단계",
                    )
                lv_answers = gr.Textbox(
                    label="정답 단어 (여러 개 가능)",
                    placeholder="사과\n바나나\n우유",
                    lines=4,
                    value="\n".join(music_stages[0]["answers"]) if music_stages else "사과",
                )
                with gr.Row():
                    lv_warm_btn = gr.Button("⚡ 준비", variant="primary")
                    lv_reset_btn = gr.Button("↺ 초기화")
                lv_window = gr.Slider(
                    1.0, 4.0, value=2.5, step=0.1,
                    label="ASR 창 (초)",
                    interactive=not bool(music_stages),
                    info="Unity 기본값은 2.5초입니다. 부족한 앞부분은 무음으로 "
                         "채워 바로 검사합니다. 길이를 바꾸면 다음 입력부터 "
                         "음성과 연속 판정을 초기화합니다.",
                )
                lv_consecutive = gr.Slider(
                    1, 6, value=2, step=1,
                    label="기본 연속 횟수 (단어 설정이 우선)" if music_stages else "확정에 필요한 연속 횟수",
                    info="발표 모드의 12개 단어는 저장된 단어별 횟수를 사용합니다." if music_stages
                         else "낮추면 빨리 반응하지만 오검출 가능성이 커집니다.",
                    interactive=not bool(music_stages),
                )
                lv_audio = gr.Audio(
                    sources=["microphone"],
                    streaming=True,
                    type="numpy",
                    label="🔴 마이크 (누르면 바로 시작)",
                )
                if music_stages:
                    file_audio = gr.Audio(sources=["upload"], type="numpy",
                                          label="녹음 파일 테스트 (종료 후 무음 1초 포함)")
                    file_button = gr.Button("음성 파일로 테스트")

            with gr.Column(scale=2):
                lv_status = gr.HTML()
                lv_feedback = gr.HTML(pronunciation_feedback_html(None))
                with gr.Accordion("확정 결과 데이터 (웹·Unity 공통 형식)", open=False):
                    lv_confirmation = gr.JSON(label="PronunciationFeedback")
                lv_thresholds = gr.Markdown()
                lv_plot = gr.LinePlot(
                    x="시각", y="점수",
                    x_title="시각 (초)",
                    y_title="점수",
                    y_lim=[0, 1],
                    height=260,
                    label="시간에 따른 점수",
                )
                lv_runs = gr.Markdown()
                lv_ipa = gr.Markdown()
                with gr.Accordion("프레임 상세", open=False):
                    lv_table = gr.Dataframe(
                        headers=["시각(s)", "연속", "선택 후보", "점수",
                                 "임계값"],
                        datatype=["number", "number", "str", "number",
                                  "number"],
                        interactive=False,
                        wrap=True,
                    )

        lv_warm_btn.click(
            fn=live_warmup,
            inputs=[lv_answers, lv_language_radio],
            outputs=[lv_status, lv_thresholds],
        )
        lv_reset_btn.click(
            fn=live_reset,
            outputs=[live_state, lv_status, lv_thresholds, lv_plot,
                     lv_runs, lv_ipa, lv_table, lv_feedback, lv_confirmation],
        )
        lv_audio.stream(
            fn=live_step,
            inputs=[lv_audio, live_state, lv_answers, lv_language_radio,
                    lv_window, lv_consecutive],
            outputs=[live_state, lv_status, lv_thresholds, lv_plot,
                     lv_runs, lv_ipa, lv_table, lv_feedback, lv_confirmation, lv_audio],
            stream_every=LIVE_HOP_S,
            time_limit=60,
            show_progress="hidden",
        )
        lv_audio.start_recording(
            fn=live_reset,
            outputs=[live_state, lv_status, lv_thresholds, lv_plot,
                     lv_runs, lv_ipa, lv_table, lv_feedback, lv_confirmation],
        )
        if music_stages:
            common_outputs = [live_state, lv_status, lv_thresholds, lv_plot,
                              lv_runs, lv_ipa, lv_table, lv_feedback,
                              lv_confirmation, lv_audio]
            lv_stage.change(select_music_stage, inputs=[lv_stage],
                            outputs=[lv_answers, *common_outputs])
            file_button.click(lambda: gr.update(recording=False), outputs=[lv_audio], api_name=False).then(
                replay_file,
                inputs=[file_audio, lv_answers, lv_language_radio, lv_window, lv_consecutive],
                outputs=common_outputs[:-1],
            )

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="로컬 포트 번호 (기본 7860)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Gradio 터널로 임시 공개 URL 노출",
    )
    parser.add_argument(
        "--matrix",
        type=Path,
        default=PROJECT_ROOT
        / "shared"
        / "confusion_matrices"
        / "ko_child_v1.json",
        help="Confusion matrix JSON 경로",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="HuggingFace ASR 모델 ID (기본: 프로젝트 기본값)",
    )
    parser.add_argument(
        "--targets",
        type=Path,
        default=None,
        help="정답 카탈로그 JSON. 주면 카탈로그에 있는 단어는 출하된 "
             "임계값과 확정 정책을 그대로 씁니다 (안 주면 전부 즉석 측정)",
    )
    parser.add_argument("--music-demo", action="store_true",
                        help="악기놀이 발표용 12단어 설정, 9단계 선택, 녹음 파일 재생")
    args = parser.parse_args()

    if args.music_demo:
        args.targets = PROJECT_ROOT / "shared/targets_music_demo.json"
        args.matrix = PROJECT_ROOT / "shared/confusion_matrices/ko_child_v2.json"
        args.model = args.model or str(PROJECT_ROOT / "python/tools/child_finetune/runs/child/best")
        if not args.targets.exists():
            parser.error("먼저 python -m python.build.build_music_presentation 을 실행하세요.")

    print("실시간 웹 UI 시작 중... 모델은 준비 버튼을 누르면 로드됩니다.")
    # Live candidates come from the textbox; no prebuilt catalog is needed.
    tester = AudioTester(
        targets_path=args.targets, matrix_path=args.matrix,
        model_name=args.model
    )

    import gradio as gr

    app = build_app(tester)
    app.launch(
        server_name="127.0.0.1",
        server_port=args.port,
        share=args.share,
        show_error=True,
        inbrowser=True,
        theme=gr.themes.Soft(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
