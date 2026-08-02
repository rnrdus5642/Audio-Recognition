"""한국어 음소 매칭 파이프라인용 Gradio 웹 UI.

로컬 웹 앱을 실행해서 다음을 할 수 있습니다:
  * 브라우저 마이크로 녹음 (또는 오디오 파일 업로드)
  * words.csv 에서 정답 단어 선택 (또는 직접 입력)
  * 통과/실패 배너, ASR 한글, IPA 음소 분해, 후보 점수 표,
    다른 세그먼트 통과 위험 경고를 한눈에 확인
  * 오디오 재생
  * 녹음한 결과를 recordings/ 폴더에 오디오 + JSON 메타로 저장

사용법:
    python -m python.tools.web_test
    python -m python.tools.web_test --port 7860 --share

처음 실행 시 HuggingFace 모델(~1.2GB)을 다운로드하므로 1분 정도
걸릴 수 있습니다. 이후 인식은 캐시된 모델을 재사용합니다.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import io
import json
import sys
from pathlib import Path
from typing import Any, Optional

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

from python.tools.test_real_audio import AudioTester, TestResult


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECORDINGS = PROJECT_ROOT / "recordings"

SAMPLE_RATE = 16_000


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------


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


def _audio_stats(audio: np.ndarray) -> dict[str, Any]:
    if len(audio) == 0:
        return {"duration_s": 0.0, "rms": 0.0, "peak": 0.0, "trailing_ms": 0}
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    peak = float(np.max(np.abs(audio)))
    threshold = 10 ** (-40 / 20)
    last_loud = len(audio)
    for i in range(len(audio) - 1, -1, -1):
        if abs(audio[i]) >= threshold:
            last_loud = i
            break
    trailing_ms = int((len(audio) - last_loud) / SAMPLE_RATE * 1000)
    return {
        "duration_s": round(len(audio) / SAMPLE_RATE, 3),
        "rms": round(rms, 4),
        "peak": round(peak, 4),
        "trailing_ms": trailing_ms,
    }


# ---------------------------------------------------------------------------
# Phoneme diff (best-effort visualisation)
# ---------------------------------------------------------------------------


def _phoneme_diff_html(
    user: list[str],
    target: list[str],
    alignment: list[tuple[str, str, str]] | None = None,
    window_start: int = 0,
    window_end: int | None = None,
) -> str:
    """Side-by-side aligned diff as HTML.

    If `alignment` is provided, it is used directly (this is the
    backtrace from substring_edit_distance and is consistent with the
    confusion-matrix-weighted scoring). The optional `window_start` /
    `window_end` are used to show user phonemes OUTSIDE the matched
    window in a dimmed style.
    """
    if not user and not target:
        return "<i>(음소 없음)</i>"

    if window_end is None:
        window_end = len(user)

    if alignment is None:
        alignment = []

    # Header row: show full user IPA with prefix/suffix dimmed
    prefix_cells = []
    cell_base = (
        "display:inline-block;min-width:32px;padding:4px 8px;"
        "margin:2px;border-radius:4px;text-align:center;"
        "font-family:'Consolas','Courier New',monospace;font-size:16px;"
    )
    for u in user[:window_start]:
        prefix_cells.append(
            f"<span style=\"{cell_base}background:#374151;"
            "color:#9ca3af;opacity:0.55;\">"
            f"{u}</span>"
        )
    suffix_cells = []
    for u in user[window_end:]:
        suffix_cells.append(
            f"<span style=\"{cell_base}background:#374151;"
            "color:#9ca3af;opacity:0.55;\">"
            f"{u}</span>"
        )
    noise_label = ""
    n_noise = len(user[:window_start]) + len(user[window_end:])
    if n_noise > 0:
        noise_label = (
            f" <span style='color:#9ca3af;font-size:12px;'>"
            f"(흐리게 = 윈도우 밖 잡음, {n_noise}개 음소)</span>"
        )

    user_cells, target_cells, op_cells = [], [], []
    for u, t, op in alignment:
        if op == "match":
            color = "#22c55e"
            sym = "✓"
        elif op == "sub":
            color = "#f97316"
            sym = "≠"
        elif op == "ins":
            color = "#ef4444"
            sym = "+"
        else:  # del
            color = "#ef4444"
            sym = "−"
        cell_style = (
            "display:inline-block;min-width:32px;padding:4px 8px;"
            "margin:2px;border-radius:4px;text-align:center;"
            "font-family:'Consolas','Courier New',monospace;font-size:16px;"
        )
        user_cells.append(
            f"<span style=\"{cell_style}background:#1f2937;color:#f3f4f6;\">"
            f"{u or '·'}</span>"
        )
        target_cells.append(
            f"<span style=\"{cell_style}background:#1f2937;color:#f3f4f6;\">"
            f"{t or '·'}</span>"
        )
        op_cells.append(
            f"<span style=\"{cell_style}background:{color};color:white;"
            "font-weight:bold;\">"
            f"{sym}</span>"
        )
    full_user_row = (
        "".join(prefix_cells) + "".join(user_cells) + "".join(suffix_cells)
    )
    # Diff row only spans the matched window; pad with placeholders so
    # columns roughly align with the user row.
    diff_row = (
        "".join(
            f"<span style=\"{cell_base}background:transparent;opacity:0;\">·</span>"
            for _ in prefix_cells
        )
        + "".join(op_cells)
        + "".join(
            f"<span style=\"{cell_base}background:transparent;opacity:0;\">·</span>"
            for _ in suffix_cells
        )
    )
    target_row = (
        "".join(
            f"<span style=\"{cell_base}background:transparent;opacity:0;\">·</span>"
            for _ in prefix_cells
        )
        + "".join(target_cells)
        + "".join(
            f"<span style=\"{cell_base}background:transparent;opacity:0;\">·</span>"
            for _ in suffix_cells
        )
    )

    return (
        "<div style='line-height:1.4;'>"
        f"<div><b>사용자:</b> {full_user_row}{noise_label}</div>"
        f"<div><b>차이:</b> {diff_row}</div>"
        f"<div><b>정답:</b> {target_row}</div>"
        "</div>"
    )


# ---------------------------------------------------------------------------
# Result formatters
# ---------------------------------------------------------------------------


def _status_html(result: TestResult) -> str:
    overall = result.overall_pass
    if overall is None:
        # Probe mode
        return (
            "<div style='padding:16px;border-radius:8px;background:#374151;"
            "color:#f3f4f6;'><h2 style='margin:0;'>🔍 ASR 확인 모드</h2>"
            f"<p style='margin:4px 0;'>ASR 결과만 표시 — 정답 매칭은 하지 않음.</p>"
            "</div>"
        )
    best = result.best
    if overall:
        return (
            "<div style='padding:16px;border-radius:8px;background:#16a34a;"
            "color:white;'>"
            f"<h2 style='margin:0;'>✓ 통과</h2>"
            f"<p style='margin:4px 0;font-size:18px;'>"
            f"<b>{best.text}</b>(으)로 매칭 "
            f"(점수 <b>{best.score:.3f}</b>, 임계값 {best.threshold:.2f})"
            "</p></div>"
        )
    if best is None:
        return (
            "<div style='padding:16px;border-radius:8px;background:#dc2626;"
            "color:white;'><h2 style='margin:0;'>✗ 실패</h2>"
            "<p style='margin:4px 0;'>매칭할 후보가 없습니다.</p>"
            "</div>"
        )
    if best.text != result.target_text:
        return (
            "<div style='padding:16px;border-radius:8px;background:#dc2626;"
            "color:white;'>"
            f"<h2 style='margin:0;'>✗ 실패 — 다른 단어로 인식됨</h2>"
            f"<p style='margin:4px 0;font-size:16px;'>"
            f"의도한 단어: <b>{result.target_text}</b>, "
            f"매칭된 단어: <b>{best.text}</b> "
            f"(점수 {best.score:.3f})</p></div>"
        )
    return (
        "<div style='padding:16px;border-radius:8px;background:#dc2626;"
        "color:white;'>"
        f"<h2 style='margin:0;'>✗ 실패 — 임계값 미달</h2>"
        f"<p style='margin:4px 0;font-size:16px;'>"
        f"정답({best.text})은 골랐지만 점수 "
        f"{best.score:.3f}이 임계값 {best.threshold:.2f}보다 낮음</p></div>"
    )


def _streaming_status_html(
    hit,
    fire_time: float | None,
    final_streak: int,
    consecutive: int,
    hop: float,
) -> str:
    """Banner for the continuous-listening tab."""
    if hit is not None:
        return (
            "<div style='padding:16px;border-radius:8px;background:#16a34a;"
            "color:white;'>"
            "<h2 style='margin:0;'>✓ 확정</h2>"
            f"<p style='margin:4px 0;font-size:18px;'>"
            f"<b>{hit.result.target_text}</b> — {fire_time:.1f}초 지점 "
            f"({hit.frames}번째 프레임, 점수 {hit.result.score:.3f})</p>"
            "<p style='margin:4px 0;font-size:14px;opacity:.9;'>"
            "실제 VR에서는 이 시점에 녹음을 멈추고 다음으로 넘어갑니다.</p>"
            "</div>"
        )
    if final_streak:
        return (
            "<div style='padding:16px;border-radius:8px;background:#d97706;"
            "color:white;'>"
            "<h2 style='margin:0;'>△ 확정 실패 — 연속 확인 부족</h2>"
            f"<p style='margin:4px 0;font-size:16px;'>"
            f"임계값을 넘었지만 최장 연속 {final_streak}회에 그쳤습니다 "
            f"(확정에 {consecutive}회 필요).</p>"
            "<p style='margin:4px 0;font-size:14px;opacity:.9;'>"
            f"아래 구간 요약에서 어디서 끊겼는지 확인하세요. 정답 직후 "
            f"녹음이 끝나도 확정되지 않으니 말한 뒤 "
            f"{consecutive * hop:.1f}초는 더 녹음하세요.</p></div>"
        )
    return (
        "<div style='padding:16px;border-radius:8px;background:#dc2626;"
        "color:white;'><h2 style='margin:0;'>✗ 미검출</h2>"
        "<p style='margin:4px 0;'>어느 프레임에서도 임계값을 넘지 "
        "못했습니다.</p></div>"
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


def _streak_summary_md(points: list[dict], consecutive: int) -> str:
    """Why the run did (or did not) confirm.

    The chart shows the score crossing its threshold, but confirmation
    also needs `consecutive` frames in a row won by the SAME answer.
    Listing the above-threshold runs makes the difference visible -
    otherwise a chart full of peaks looks like it should have passed.
    """
    if not points:
        return ""

    runs = _passing_runs(points)
    n_over = sum(1 for p in points if p["score"] >= p["threshold"])
    if not runs:
        return (
            f"임계값을 넘은 프레임이 없습니다 "
            f"(전체 {len(points)}프레임)."
        )

    longest = max(len(r) for r in runs)
    lines = [
        f"**임계값 초과 구간** — 전체 {len(points)}프레임 중 {n_over}개 초과, "
        f"최장 연속 **{longest}회** (확정에 {consecutive}회 필요)"
    ]
    for r in runs:
        span = (f"{r[0]['t']:.1f}초" if len(r) == 1
                else f"{r[0]['t']:.1f}~{r[-1]['t']:.1f}초")
        mark = " ✓" if len(r) >= consecutive else ""
        lines.append(
            f"- {span} · **{r[0]['word']}** · {len(r)}프레임"
            f" (점수 {min(p['score'] for p in r):.2f}"
            f"~{max(p['score'] for p in r):.2f}){mark}"
        )
    if longest < consecutive:
        lines.append(
            f"\n➜ 넘기는 했지만 **연속으로 이어지지 않아** 확정되지 "
            f"않았습니다. 정답을 조금 더 길게(또는 한 번 더) 말하면 "
            f"연속 {consecutive}회를 채웁니다."
        )
    return "\n".join(lines)


def _thresholds_md(candidates: list[dict]) -> str:
    """One-line reference of each answer's pass mark.

    Thresholds are derived from phoneme count, so showing the count
    explains why they differ.
    """
    if not candidates:
        return ""
    parts = [
        f"**{c['text']}** `{c['threshold']:.2f}`"
        f" <sub>({len(c['phonemes'])}음소)</sub>"
        for c in candidates
    ]
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


def _audio_stats_html(stats: dict) -> str:
    warns = []
    if stats["peak"] < 0.02 and stats["duration_s"] > 0:
        warns.append("⚠ 너무 조용함")
    elif stats["peak"] > 0.98:
        warns.append("⚠ 클리핑 (음량 과다)")
    if stats["trailing_ms"] < 80 and stats["duration_s"] > 0.3:
        warns.append("⚠ 끝 묵음 짧음 (마지막 소리 잘렸을 수 있음)")
    warn_str = (
        f" <span style='color:#dc2626;'>{', '.join(warns)}</span>"
        if warns
        else ""
    )
    return (
        f"<div style='font-family:monospace;font-size:13px;color:#6b7280;'>"
        f"오디오: {stats['duration_s']:.2f}초 · "
        f"RMS={stats['rms']:.3f} · 최댓값={stats['peak']:.3f} · "
        f"끝묵음={stats['trailing_ms']}ms{warn_str}</div>"
    )


# ---------------------------------------------------------------------------
# Target catalog
# ---------------------------------------------------------------------------


def build_target_catalog(
    tester: AudioTester,
) -> list[tuple[str, str, str, str]]:
    """Return [(label, segment_id, target_text, language)] for the dropdown."""
    out = []
    for seg in tester.targets["segments"]:
        lang = seg.get("language", "ko")
        for ans in seg["answers"]:
            label = f"{ans['text']}  ·  {ans['id']}  ·  [{seg['id']}]"
            out.append((label, seg["id"], ans["text"], lang))
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


def _language_label(code: str) -> str:
    return LANGUAGE_DISPLAY_NAMES.get(code, code.upper())


# ---------------------------------------------------------------------------
# Save take
# ---------------------------------------------------------------------------


def save_take(
    audio: np.ndarray,
    target_text: Optional[str],
    target_segment: Optional[str],
    result: TestResult,
    out_dir: Path,
) -> Path:
    import soundfile as sf

    out_dir.mkdir(parents=True, exist_ok=True)
    session_dir = out_dir / f"web_{_dt.datetime.now().strftime('%Y%m%d')}"
    session_dir.mkdir(parents=True, exist_ok=True)

    ts = _dt.datetime.now().strftime("%H%M%S")
    safe_label = (target_text or "probe").replace("/", "_")
    stem = f"{ts}_{safe_label}"
    wav_path = session_dir / f"{stem}.wav"
    json_path = session_dir / f"{stem}.json"

    sf.write(wav_path, audio, SAMPLE_RATE, subtype="PCM_16")
    stats = _audio_stats(audio)
    payload = {
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "audio_file": wav_path.name,
        "audio_stats": stats,
        "target": {"text": target_text, "segment_id": target_segment},
        "result": {
            "hangul": result.hangul,
            "user_ipa": result.user_ipa,
            "best": (
                {
                    "answer_id": result.best.answer_id,
                    "text": result.best.text,
                    "score": round(result.best.score, 4),
                    "passed": result.best.would_pass,
                }
                if result.best
                else None
            ),
            "overall_pass": result.overall_pass,
            "candidates": [
                {
                    "answer_id": c.answer_id,
                    "text": c.text,
                    "phonemes": c.phonemes,
                    "distance": round(c.distance, 4),
                    "score": round(c.score, 4),
                    "threshold": c.threshold,
                    "would_pass": c.would_pass,
                    "is_best": c.is_best,
                }
                for c in result.candidates
            ],
            "cross_segment_best": (
                {
                    "answer_id": result.cross_segment_best.answer_id,
                    "text": result.cross_segment_best.text,
                    "score": round(result.cross_segment_best.score, 4),
                    "would_pass": result.cross_segment_best.would_pass,
                }
                if result.cross_segment_best
                else None
            ),
        },
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return wav_path


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------


def build_app(tester: AudioTester, recordings_dir: Path):
    import gradio as gr

    # Languages that have a registered recognizer (G2P + ASR).
    from python.runtime.recognizer import available_languages as available_recognizers

    target_languages = tester.available_target_languages()
    recognized = set(available_recognizers())
    # Show recognizer languages first, fall back to target.json's languages.
    usable_languages = sorted(recognized) or target_languages or ["ko"]

    # 모드 라벨 (UI에 표시되는 이름)
    MODE_CUSTOM = "직접 입력"
    MODE_PROBE = "ASR 결과만 보기"

    def run_recognition(
        audio_in,
        custom_target,
        mode,
        language,
        history,
    ):
        # Resolve target
        custom = mode == MODE_CUSTOM
        probe = mode == MODE_PROBE
        scan_all = False  # 더 이상 사용 안 함 (UI에서 제거)
        lang = language or "ko"

        if probe:
            target_text, segment_id = None, None
        elif custom:
            target_text = (custom_target or "").strip()
            if not target_text:
                return (
                    "<div style='padding:8px;color:#dc2626;'>"
                    "단어 입력 칸에 단어를 입력하세요.</div>",
                    "", "", "", history, gr.update(), "",
                )
            segment_id = None
        else:
            # 어떤 모드도 매치되지 않음 (방어적)
            return (
                "<div style='padding:8px;color:#dc2626;'>"
                "모드를 선택하세요.</div>",
                "", "", [], "", "", history, gr.update(), "",
            )

        audio = _normalize_audio_input(audio_in)
        if len(audio) == 0:
            return (
                "<div style='padding:8px;color:#dc2626;'>"
                "녹음된 오디오가 없습니다. 마이크 아이콘을 눌러 먼저 녹음하세요.</div>",
                "", "", [], "", "", history, gr.update(), "",
            )

        stats = _audio_stats(audio)

        try:
            result = tester.test_array(
                audio=audio,
                target_text=target_text,
                segment_id=segment_id,
                custom_target=custom,
                scan_all=scan_all,
                language=lang,
            )
        except Exception as e:
            return (
                f"<div style='padding:8px;color:#dc2626;'>오류: {e}</div>",
                "", "", [], "", "", history, gr.update(), "",
            )

        # Build outputs
        status = _status_html(result)
        # The ASR section's first line is language-specific (the raw text
        # produced by the ASR — Hangul for Korean, etc.).
        raw_label = "한글" if lang == "ko" else "텍스트"
        hangul_md = (
            f"### ASR 인식 결과 ({_language_label(lang)})\n"
            f"**{raw_label}:** `{result.hangul or '(빈 출력)'}`\n\n"
            f"**IPA 음소:** `[{', '.join(result.user_ipa) or '(빈 출력)'}]`"
        )
        diff_html = ""
        if result.best is not None:
            diff_html = _phoneme_diff_html(
                result.user_ipa,
                result.best.phonemes,
                alignment=result.best.alignment,
                window_start=result.best.window_start,
                window_end=result.best.window_end,
            )
        stats_html = _audio_stats_html(stats)

        # Append to history (in-memory list for the session)
        history = (history or []).copy()
        history.insert(
            0,
            {
                "time": _dt.datetime.now().strftime("%H:%M:%S"),
                "target": target_text or "(ASR만)",
                "hangul": result.hangul,
                "ipa": result.user_ipa,
                "score": (
                    round(result.best.score, 3)
                    if result.best is not None
                    else None
                ),
                "pass": result.overall_pass,
            },
        )
        history = history[:20]

        history_md = _format_history(history)

        # Stash audio + result for the save button
        save_state = {
            "audio": audio.tolist(),
            "target_text": target_text,
            "segment_id": segment_id,
        }

        return (
            status,
            stats_html,
            hangul_md,
            diff_html,
            history,
            save_state,
            history_md,
        )

    def run_streaming(
        audio_in,
        answers_text,
        language,
        window_s,
        hop_s,
        consecutive,
        progress=gr.Progress(),
    ):
        """Replay a recording through the continuous-listening pipeline.

        Same decision logic the VR app will run: re-recognise the
        trailing window every hop, score against the candidates, commit
        once one of them wins `consecutive` frames in a row.
        """
        from python.runtime.audio import rolling_windows
        from python.runtime.matching import Matcher, StreamingMatcher

        lang = language or "ko"
        words = [
            w.strip()
            for w in (answers_text or "").replace(",", "\n").splitlines()
            if w.strip() and not w.strip().startswith("#")
        ]
        empty = ("", _score_plot_df([]), "", _ipa_md([], 0, None, None),
                 [], "")
        if not words:
            return (
                "<div style='padding:8px;color:#dc2626;'>정답 단어를 "
                "한 줄에 하나씩(또는 쉼표로) 입력하세요.</div>", *empty,
            )

        audio = _normalize_audio_input(audio_in)
        if len(audio) == 0:
            return (
                "<div style='padding:8px;color:#dc2626;'>녹음된 오디오가 "
                "없습니다.</div>", *empty,
            )

        try:
            # `_make_custom_target` hands out a fixed id, but the streak
            # logic keys on target_id - identical ids would let two
            # different words take turns winning and still confirm.
            candidates = []
            for i, w in enumerate(words):
                cand = dict(tester._make_custom_target(w, lang)[1])
                cand["id"] = f"custom_{i}"
                candidates.append(cand)
        except Exception as e:
            return (
                f"<div style='padding:8px;color:#dc2626;'>G2P 실패: {e}</div>",
                *empty,
            )

        matcher = Matcher.for_streaming(tester._matrix_for(lang))
        sm = StreamingMatcher(matcher, candidates, consecutive=int(consecutive))
        rec = tester._recognizer_for(lang)

        frames = list(rolling_windows(audio, float(window_s), float(hop_s)))
        rows: list[list] = []
        points: list[dict] = []
        hit = None
        fire_time = None
        last_ipa_md = "**채점 대상 IPA:** _(없음)_"
        for i, (t, window) in enumerate(frames):
            if progress is not None:
                progress((i + 1) / len(frames), desc=f"{t:.1f}초 인식 중")
            _hangul, ipa = rec.recognize_with_text(window)
            hit = sm.push(ipa)
            best = matcher.best_match(ipa, candidates)
            threshold = next(
                (c["threshold"] for c in candidates
                 if c["id"] == best.target_id),
                0.0,
            )
            rows.append([
                round(t, 1),
                sm.streak,
                best.target_text or "—",
                round(best.score, 3),
                threshold,
            ])
            points.append({
                "_i": i,
                "t": round(t, 2),
                "score": round(best.score, 3),
                "threshold": threshold,
                "word": best.target_text if best.passed else None,
            })
            scored, dropped = matcher.context_slice(
                ipa, best.target_phonemes or []
            )
            last_ipa_md = _ipa_md(
                scored, dropped,
                (best.window_start - dropped, best.window_end - dropped),
                best.target_text,
            )
            if hit:
                fire_time = t
                break

        note = (
            f"창 {window_s}초 / hop {hop_s}초 / 연속 {int(consecutive)}회 · "
            f"skip_cost {matcher.skip_cost} · coverage {matcher.coverage} · "
            f"문맥 최근 {matcher.context_mult}×정답음소수+3 · "
            f"후보 {', '.join(words)}"
        )
        if hit is None and len(frames) > len(rows):
            note += "  (중단됨)"
        # The banner should report the best streak the session reached,
        # not the streak left on the final frame - those differ whenever
        # the answer was detected earlier and then dropped out.
        best_streak = max((len(r) for r in _passing_runs(points)), default=0)
        return (
            _streaming_status_html(
                hit, fire_time, best_streak, int(consecutive), float(hop_s)
            ),
            _thresholds_md(candidates),
            _score_plot_df(points),
            _streak_summary_md(points, int(consecutive)),
            last_ipa_md,
            rows,
            note,
        )

    # ------------------------------------------------------------------
    # Live (microphone-streaming) continuous listening
    # ------------------------------------------------------------------

    LIVE_HOP_S = 0.5  # must match `stream_every` in the wiring below

    def _live_candidates(answers_text, lang):
        words = [
            w.strip()
            for w in (answers_text or "").replace(",", "\n").splitlines()
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
                _streak_summary_md(st.get("points", []), int(consecutive)),
                st.get("ipa_md", _ipa_md([], 0, None, None)),
                st.get("rows", []),
                gr.update(),
            )

        state = state or {}
        # Already confirmed, or nothing to process - leave the view as is.
        if state.get("done") or new_chunk is None:
            return unchanged(state)

        lang = language or "ko"
        key = f"{lang}|{answers_text}|{consecutive}"
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
                    matcher, cands, consecutive=int(consecutive)
                ),
                "cands": cands,
                "buf": np.zeros(0, dtype=np.float32),
                "rows": [],
                "points": [],
                "elapsed": 0.0,
            }

        chunk = _normalize_audio_input(new_chunk)
        if len(chunk) == 0:
            return unchanged(state)

        keep = int(float(window_s) * SAMPLE_RATE)
        state["buf"] = np.concatenate([state["buf"], chunk])[-keep:]
        state["elapsed"] += len(chunk) / SAMPLE_RATE

        rec = tester._recognizer_for(lang)
        _hangul, ipa = rec.recognize_with_text(state["buf"])

        sm = state["sm"]
        hit = sm.push(ipa)
        best = sm.matcher.best_match(ipa, state["cands"])
        threshold = next(
            (c["threshold"] for c in state["cands"]
             if c["id"] == best.target_id), 0.0
        )
        t = round(state["elapsed"], 1)
        state["rows"] = ([[
            t, sm.streak, best.target_text or "—",
            round(best.score, 3), threshold,
        ]] + state["rows"])[:20]
        state["points"].append({
            "_i": len(state["points"]),
            "t": t, "score": round(best.score, 3), "threshold": threshold,
            "word": best.target_text if best.passed else None,
        })
        scored, dropped = sm.matcher.context_slice(
            ipa, best.target_phonemes or []
        )
        state["ipa_md"] = _ipa_md(
            scored, dropped,
            (best.window_start - dropped, best.window_end - dropped),
            best.target_text,
        )
        plot = _score_plot_df(state["points"])

        thresholds = _thresholds_md(state["cands"])
        runs = _streak_summary_md(state["points"], int(consecutive))

        if hit:
            state["done"] = True
            state["status"] = _streaming_status_html(
                hit, state["elapsed"], sm.streak,
                int(consecutive), LIVE_HOP_S
            )
            # Stop the mic - this is the "answer accepted, move on" moment.
            return (state, state["status"], thresholds, plot, runs,
                    state["ipa_md"], state["rows"],
                    gr.update(recording=False))

        state["status"] = (
            "<div style='padding:16px;border-radius:8px;background:#1f2937;"
            "color:#f3f4f6;'><h2 style='margin:0;'>🎧 듣는 중…</h2>"
            f"<p style='margin:4px 0;font-size:16px;'>{state['elapsed']:.1f}초 · "
            f"연속 <b>{sm.streak}</b>/{int(consecutive)} · "
            f"최고 {best.target_text or '—'} "
            f"{best.score:.3f} / {threshold:.2f}</p></div>"
        )
        return (state, state["status"], thresholds, plot, runs,
                state["ipa_md"], state["rows"], gr.update())

    def do_save(save_state):
        if not save_state or not save_state.get("audio"):
            return "<div style='color:#dc2626;'>저장할 데이터가 없습니다. 먼저 녹음 + 인식하세요.</div>"
        audio = np.asarray(save_state["audio"], dtype=np.float32)
        target_text = save_state.get("target_text")
        target_segment = save_state.get("segment_id")
        # Re-run to get a fresh result (cheap on cached model)
        try:
            result = tester.test_array(
                audio=audio,
                target_text=target_text,
                segment_id=target_segment,
                custom_target=False,
            )
        except Exception as e:
            return f"<div style='color:#dc2626;'>저장 실패: {e}</div>"
        path = save_take(audio, target_text, target_segment, result, recordings_dir)
        rel = path.relative_to(PROJECT_ROOT)
        return (
            f"<div style='padding:8px;background:#dcfce7;border-radius:6px;"
            f"color:#166534;'>저장됨: <code>{rel}</code></div>"
        )

    from python.tools import _web_builder

    with gr.Blocks(title="음소 매칭 시스템") as app:
        gr.Markdown(
            "# 🎤 음소 매칭 시스템\n"
            "발음 매칭 테스트 + 정답 데이터 빌드를 한 곳에서."
        )

        save_state = gr.State({})
        history_state = gr.State([])

      # ─── Tab 컨텍스트: 시각적 들여쓰기는 유지하기 위해 헬퍼 사용
      # (Gradio Tab은 with-context. 기존 코드 재들여쓰기를 피하려고
      #  로컬 함수로 본문을 묶음.)

        # 발음 테스트 탭의 본문은 아래 with 블록 안 (원래 코드 유지).
        # 들여쓰기 변경을 최소화하기 위해 Tab을 with 블록 직전에 진입.
        _tab_test = gr.Tab("🎤 발음 테스트")
        _tab_test.__enter__()
        with gr.Row():
            # ---------- 왼쪽: 입력 ----------
            with gr.Column(scale=1):
                language_choices = [
                    (_language_label(lang), lang)
                    for lang in usable_languages
                ]
                language_radio = gr.Radio(
                    choices=language_choices,
                    value=usable_languages[0],
                    label="언어",
                    interactive=True,
                    visible=len(usable_languages) > 1,
                )

                mode = gr.Radio(
                    choices=[MODE_CUSTOM, MODE_PROBE],
                    value=MODE_CUSTOM,
                    label="모드",
                    interactive=True,
                )

                custom_target = gr.Textbox(
                    label="정답 단어",
                    placeholder="예: 사과",
                    visible=True,
                )

                def _on_mode_change(m):
                    return gr.update(visible=(m == MODE_CUSTOM))

                mode.change(
                    fn=_on_mode_change,
                    inputs=mode,
                    outputs=custom_target,
                )

                audio_input = gr.Audio(
                    sources=["microphone", "upload"],
                    type="numpy",
                    label="🎤 녹음 또는 파일 업로드",
                )

                with gr.Row():
                    submit_btn = gr.Button(
                        "🔍 인식하기", variant="primary", scale=2
                    )
                    save_btn = gr.Button("💾 저장", scale=1)
                save_msg = gr.HTML()

            # ---------- 오른쪽: 결과 ----------
            with gr.Column(scale=2):
                status_html = gr.HTML()
                audio_stats_html = gr.HTML()

                hangul_md = gr.Markdown()

                with gr.Accordion("음소 정렬 (사용자 vs 정답)", open=True):
                    diff_html = gr.HTML()

                with gr.Accordion("이번 세션 기록", open=False):
                    history_md_view = gr.Markdown()

        # 발음 테스트 탭 끝
        _tab_test.__exit__(None, None, None)

        # ============================================================
        #  Tab 2: 🔁 연속 청취 (VR 흐름)
        # ============================================================
        with gr.Tab("🔁 연속 청취"):
            gr.Markdown(
                "VR 흐름을 재현합니다. 녹음 전체를 한 번에 채점하지 않고, "
                "**최근 창을 프레임마다 다시 인식**해서 정답이 "
                "**연속 N회** 이기면 확정합니다.\n\n"
                "앞에서 딴말을 해도 됩니다 — 정답을 말하는 순간 잡힙니다. "
                "다만 **정답을 말한 뒤 잠깐 더 녹음**해야 연속 확인이 "
                "완성됩니다."
            )

            with gr.Row():
                with gr.Column(scale=1):
                    st_language_radio = gr.Radio(
                        choices=language_choices,
                        value=usable_languages[0],
                        label="언어",
                        interactive=True,
                        visible=len(usable_languages) > 1,
                    )
                    st_answers = gr.Textbox(
                        label="정답 단어 (여러 개 가능)",
                        placeholder="사과\n바나나\n우유",
                        lines=4,
                        value="사과",
                    )
                    st_audio = gr.Audio(
                        sources=["microphone", "upload"],
                        type="numpy",
                        label="🎤 녹음 또는 파일 업로드",
                    )
                    with gr.Accordion("파라미터", open=False):
                        st_window = gr.Slider(
                            1.0, 4.0, value=2.5, step=0.1,
                            label="ASR 창 (초)",
                            info="짧으면 문맥이 부족해 인식이 나빠지고, "
                                 "길면 무음이 많이 섞입니다.",
                        )
                        st_hop = gr.Slider(
                            0.2, 1.0, value=0.4, step=0.1,
                            label="채점 주기 hop (초)",
                        )
                        st_consecutive = gr.Slider(
                            1, 6, value=3, step=1,
                            label="확정에 필요한 연속 횟수",
                            info="낮추면 빨리 반응하지만 오검출이 늘어납니다. "
                                 "1이면 무관한 발화도 통과합니다.",
                        )
                    st_btn = gr.Button("🔁 연속 청취 실행", variant="primary")

                with gr.Column(scale=2):
                    st_status = gr.HTML()
                    st_thresholds = gr.Markdown()
                    st_plot = gr.LinePlot(
                        x="시각", y="점수",
                        x_title="시각 (초)",
                        y_title="점수",
                        y_lim=[0, 1],
                        height=260,
                        label="시간에 따른 점수",
                    )
                    st_runs = gr.Markdown()
                    st_ipa = gr.Markdown()
                    st_note = gr.Markdown()
                    with gr.Accordion("프레임 상세", open=False):
                        st_table = gr.Dataframe(
                            headers=["시각(s)", "연속", "최고 후보", "점수",
                                     "임계값"],
                            datatype=["number", "number", "str", "number",
                                      "number"],
                            interactive=False,
                            wrap=True,
                        )

            st_btn.click(
                fn=run_streaming,
                inputs=[st_audio, st_answers, st_language_radio,
                        st_window, st_hop, st_consecutive],
                outputs=[st_status, st_thresholds, st_plot, st_runs, st_ipa,
                         st_table, st_note],
            )

        # ============================================================
        #  Tab 3: 🔴 실시간 (마이크 스트리밍)
        # ============================================================
        with gr.Tab("🔴 실시간"):
            gr.Markdown(
                "말하는 도중 실시간으로 채점합니다. 정답이 **연속 N회** "
                "이기면 그 즉시 **녹음이 자동으로 멈춥니다** — VR에서 다음 "
                "문제로 넘어가는 시점입니다.\n\n"
                "⚠ 먼저 **준비** 버튼으로 모델을 예열하세요. 예열 없이 "
                "시작하면 첫 인식에서 1~2분 멈춥니다.\n\n"
                f"채점 주기는 {LIVE_HOP_S}초 고정입니다 "
                "(hop을 바꿔가며 비교하려면 🔁 연속 청취 탭을 쓰세요)."
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
                    lv_answers = gr.Textbox(
                        label="정답 단어 (여러 개 가능)",
                        placeholder="사과\n바나나\n우유",
                        lines=4,
                        value="사과",
                    )
                    with gr.Row():
                        lv_warm_btn = gr.Button("⚡ 준비", variant="primary")
                        lv_reset_btn = gr.Button("↺ 초기화")
                    lv_window = gr.Slider(
                        1.0, 4.0, value=2.5, step=0.1,
                        label="ASR 창 (초)",
                    )
                    lv_consecutive = gr.Slider(
                        1, 6, value=3, step=1,
                        label="확정에 필요한 연속 횟수",
                        info="1이면 무관한 발화도 통과합니다.",
                    )
                    lv_audio = gr.Audio(
                        sources=["microphone"],
                        streaming=True,
                        type="numpy",
                        label="🔴 마이크 (누르면 바로 시작)",
                    )

                with gr.Column(scale=2):
                    lv_status = gr.HTML()
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
                            headers=["시각(s)", "연속", "최고 후보", "점수",
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
                         lv_runs, lv_ipa, lv_table],
            )
            lv_audio.stream(
                fn=live_step,
                inputs=[lv_audio, live_state, lv_answers, lv_language_radio,
                        lv_window, lv_consecutive],
                outputs=[live_state, lv_status, lv_thresholds, lv_plot,
                         lv_runs, lv_ipa, lv_table, lv_audio],
                stream_every=LIVE_HOP_S,
                time_limit=60,
                show_progress="hidden",
            )
            lv_audio.start_recording(
                fn=live_reset,
                outputs=[live_state, lv_status, lv_thresholds, lv_plot,
                         lv_runs, lv_ipa, lv_table],
            )

        # ============================================================
        #  Tab 4: 📝 정답 데이터 만들기
        # ============================================================
        with gr.Tab("📝 정답 데이터 만들기"):
            gr.Markdown(
                "언어를 고르고 단어를 한 줄에 하나씩 입력하세요. "
                "G2P로 IPA를 추출해 JSON으로 내려받을 수 있습니다."
            )

            with gr.Row():
                with gr.Column(scale=1):
                    bld_language_radio = gr.Radio(
                        choices=language_choices,
                        value=usable_languages[0],
                        label="언어",
                        interactive=True,
                    )
                    bld_words_text = gr.Textbox(
                        label="단어 (한 줄에 하나)",
                        lines=12,
                        placeholder="사과\n바나나\n우유\n# '#'으로 시작하면 주석",
                    )
                    bld_extract_btn = gr.Button(
                        "🚀 추출 & 다운로드", variant="primary"
                    )
                    bld_status = gr.HTML()

                with gr.Column(scale=2):
                    bld_table = gr.Dataframe(
                        headers=["#", "단어", "IPA", "음소 수"],
                        datatype=["number", "str", "str", "number"],
                        interactive=False,
                        wrap=True,
                    )
                    bld_download_file = gr.File(
                        label="다운로드", interactive=False
                    )

            def _do_extract(language, raw_text):
                entries, errors = _web_builder.extract(language, raw_text)
                table = _web_builder.render_table(entries)
                if not entries:
                    return (
                        table,
                        _web_builder.status_error(
                            "<br>".join(errors) or "추출 결과가 없습니다."
                        ),
                        None,
                    )
                file_path = _web_builder.export_json(entries)
                msg = f"<b>{len(entries)}건 추출 완료.</b>"
                if errors:
                    msg += (
                        f"<br>실패 {len(errors)}건:<br>"
                        + "<br>".join(f"· {e}" for e in errors[:10])
                    )
                    return table, _web_builder.status_warn(msg), file_path
                return table, _web_builder.status_ok(msg), file_path

            bld_extract_btn.click(
                fn=_do_extract,
                inputs=[bld_language_radio, bld_words_text],
                outputs=[bld_table, bld_status, bld_download_file],
            )

        # ============================================================
        # 발음 테스트 탭의 와이어링
        # ============================================================
        submit_btn.click(
            fn=run_recognition,
            inputs=[
                audio_input,
                custom_target,
                mode,
                language_radio,
                history_state,
            ],
            outputs=[
                status_html,
                audio_stats_html,
                hangul_md,
                diff_html,
                history_state,
                save_state,
                history_md_view,
            ],
        )

        save_btn.click(
            fn=do_save,
            inputs=save_state,
            outputs=save_msg,
        )

    return app


def _format_history(history: list[dict]) -> str:
    if not history:
        return "_(이번 세션에서 아직 인식한 결과가 없습니다)_"
    rows = ["| 시각 | 정답 단어 | ASR 한글 | 점수 | 결과 |", "|---|---|---|---|---|"]
    for h in history:
        score = "—" if h["score"] is None else f"{h['score']:.3f}"
        passed = h.get("pass")
        icon = "✓" if passed else ("✗" if passed is False else "—")
        rows.append(
            f"| {h['time']} | {h['target']} | "
            f"{h['hangul'] or '(빈 출력)'} | {score} | {icon} |"
        )
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


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
        "--recordings-dir",
        type=Path,
        default=DEFAULT_RECORDINGS,
        help="녹음 저장 폴더 (기본 recordings/)",
    )
    parser.add_argument(
        "--targets",
        type=Path,
        default=PROJECT_ROOT / "shared" / "targets.json",
        help="targets.json 경로",
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
    args = parser.parse_args()

    print("ASR 모델 로딩 중... (최초 실행 시 ~1.2GB 다운로드, 1~2분 소요)")
    if args.model:
        tester = AudioTester(
            targets_path=args.targets,
            matrix_path=args.matrix,
            model_name=args.model,
        )
    else:
        tester = AudioTester(
            targets_path=args.targets, matrix_path=args.matrix
        )

    import gradio as gr

    app = build_app(tester, args.recordings_dir)
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
