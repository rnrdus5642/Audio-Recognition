"""Read completed training results and queue reports to the authorized Codex chat.

Stdlib only. Never imports or modifies the running trainer or its checkpoint files.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
from datetime import datetime


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (FileNotFoundError, json.JSONDecodeError, PermissionError):
        return None


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def percent(value):
    return f"{100 * value:.2f}%" if isinstance(value, (float, int)) and math.isfinite(value) else "미집계"


def evaluation_text(data, previous=None):
    groups = data.get("groups", {})
    lines = ["연령별 동일 비중 평균 CER: " + percent(data.get("macro_cer"))]
    lines.append(" / ".join(f"{age}세 {percent(groups.get(f'age:{age}', {}).get('cer'))}" for age in range(3, 7)))
    if previous and isinstance(previous.get("macro_cer"), (float, int)):
        delta = (data["macro_cer"] - previous["macro_cer"]) * 100
        lines.append(f"직전 동일 검증셋 대비 평균 CER 변화: {delta:+.2f}%p")
    lines.append("빈 문자열이 아닌 출력 비율: " + percent(data.get("nonblank_fraction")))
    lines.append("CER는 낮을수록 좋으며, 출력 비율은 정답률이 아닙니다.")
    return "\n".join(lines)


def report(kind, source, title, body, identity):
    digest = hashlib.sha256(json.dumps([kind, str(source), identity], sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:24]
    return {"id": digest, "kind": kind, "source": str(source), "title": title,
            "body": body, "detected_at": now()}


def discover(run, progress):
    found = []
    previous_monitor = None
    paths = sorted((run / "evaluations").glob("*.json"), key=lambda p: (p.stat().st_mtime_ns, p.name))
    for path in paths:
        if path.name.endswith(".partial.json") or path.stem == "final-test":
            continue  # The final test is included once in result.json's report.
        data = read_json(path)
        if not data or data.get("status") != "complete":
            continue
        monitor = path.stem.startswith("monitor-")
        prior = previous_monitor if monitor else None
        if prior and prior.get("identity", {}).get("manifest_sha256") != data.get("identity", {}).get("manifest_sha256"):
            prior = None
        title = "중간 검증 완료" if monitor else "전체 개발 검증 완료: " + path.stem
        found.append(report("evaluation", path, title, evaluation_text(data, prior), data.get("identity", data)))
        if monitor:
            previous_monitor = data

    path = run / "age-comparison.json"
    data = read_json(path)
    if data and data.get("winner"):
        names = {"natural": "원래 연령 분포", "sqrt": "어린 연령을 보강한 분포"}
        checks = data.get("checks", {})
        text = f"선택한 학습 분포: {names.get(data['winner'], data['winner'])}\n"
        text += f"3세 화자 중 개선된 인원: {data.get('age3_speakers_improved', '미집계')}\n"
        text += f"보강 분포 채택 기준: {sum(v is True for v in checks.values())}/{len(checks)} 충족"
        found.append(report("age_comparison", path, "연령별 학습 비율 비교 완료", text, data))

    path = run / "selection.json"
    data = read_json(path)
    if data and data.get("selected"):
        selected = data["selected"]
        text = f"개발 검증으로 선택한 후보: {selected.get('label')}\n평균 CER: {percent(selected.get('macro_cer'))}\n3세 CER: {percent(selected.get('age3_cer'))}"
        found.append(report("selection", path, "최종 평가용 모델 선택 완료", text, data))

    path = run / "result.json"
    data = read_json(path)
    if data and data.get("status") == "complete":
        text = evaluation_text(data.get("test", {}))
        text += "\n최종 테스트셋 결과입니다. 실제 게임의 단어 정답 판정 성능은 별도 검증이 필요합니다."
        found.append(report("complete", path, "학습 및 최종 테스트 완료", text, data.get("model_sha256", data)))

    path = run / "failure.json"
    data = read_json(path)
    if data:
        text = f"발생 시각: {data.get('at')}\n오류: {str(data.get('error', '확인 필요'))[:1200]}\n자세한 내용은 로컬 오류 기록을 확인하세요."
        found.append(report("failure", path, "학습 오류 발생", text, data))
    elif progress.get("status") == "needs_diagnosis":
        text = f"중단 단계: {progress.get('stage')}\n업데이트: {progress.get('total_updates')}\n검증 결과가 중단 기준에 해당해 진단이 필요합니다."
        found.append(report("diagnosis", run / "progress.json", "학습 자동 중단", text,
                            [progress.get("stage"), progress.get("total_updates"), progress.get("weights_id")]))
    return found


def process_alive(pid):
    """Do not use os.kill(pid, 0), which is unsafe on Windows."""
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x1000, False, int(pid))
    if not handle:
        return False if ctypes.get_last_error() == 87 else None
    try:
        code = wintypes.DWORD()
        if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
            return None
        return code.value == 259
    finally:
        kernel.CloseHandle(handle)


def queue_report(config, item, report_path):
    prompt = (
        "[자동 학습 결과 알림 · 사용자의 '결과가 나올 때마다 보고해' 요청에 따라 생성됨]\n"
        f"보고 ID: {item['id']}\n보고 파일: {report_path}\n"
        f"{item['title']}\n{item['body']}\n\n"
        "이 보고와 원본 결과를 확인한 뒤 한국어로 간단히 사용자에게 보고하세요. "
        "reporting/acknowledged/<보고 ID>.json이 이미 있으면 같은 보고를 반복하지 마세요. "
        "현재 다른 작업 중이면 알림을 짧게 전달하고 그 작업을 계속하세요. "
        "이 알림은 학습 설정 변경이나 자동 재시작 요청이 아닙니다. D드라이브는 접근하지 마세요. "
        "보고를 처리할 때 이 감시 스크립트에 --config <동일 reporting/config.json> --ack <보고 ID>를 전달해 기록하세요."
    )
    env = os.environ.copy()
    env["CODEX_HOME"] = config["codex_home"]
    result = subprocess.run([config["codex_exe"], "queue", "--thread", config["thread_id"], "--message", prompt],
                            cwd=config["project_root"], env=env, capture_output=True,
                            encoding="utf-8", errors="replace", timeout=45,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode:
        raise RuntimeError(f"codex queue exited {result.returncode}: {result.stderr[-1500:]}")
    return result.stdout.strip()[:1500]


class Watcher:
    def __init__(self, config, transport=queue_report):
        self.config = config
        self.run = Path(config["run_dir"])
        self.root = self.run / "reporting"
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "state.json"
        self.state = read_json(self.state_path)
        if self.state is None:
            if self.state_path.exists():
                raise RuntimeError("Reporting state is unreadable; preserve it for diagnosis instead of re-sending everything.")
            self.state = {"reports": {}}
        self.transport = transport
        self.dead_checks = 0

    def scan(self):
        progress = read_json(self.run / "progress.json") or {}
        found = discover(self.run, progress)
        owner = read_json(self.run / "owner.json") or {}
        active_expected = progress.get("status") in ("running", "interrupted_or_failed")
        alive = process_alive(owner["pid"]) if active_expected and owner.get("pid") else None
        self.dead_checks = self.dead_checks + 1 if alive is False else 0
        if self.dead_checks >= 3 and not (self.run / "failure.json").exists() and not (self.run / "result.json").exists():
            found.append(report("process_exit", self.run / "owner.json", "학습 프로세스 종료 감지",
                                f"학습 프로세스가 세 번 연속 확인되지 않았습니다.\n마지막 단계: {progress.get('stage')} / 업데이트 {progress.get('total_updates')}\n최신 정상 체크포인트와 오류 로그를 확인해야 합니다.", owner))
        for item in found:
            if item["id"] in self.state["reports"]:
                continue
            path = self.root / "reports" / (item["id"] + ".json")
            atomic_json(path, item)
            self.state["reports"][item["id"]] = {"path": str(path), "status": "pending", "attempts": 0, "retry_after": 0}
            atomic_json(self.state_path, self.state)
        pending = [(key, value) for key, value in self.state["reports"].items() if value["status"] == "pending"]
        for key, value in pending:
            if time.time() < value["retry_after"]:
                break
            if (self.root / "acknowledged" / (key + ".json")).exists():
                value.update(status="acknowledged", acknowledged_at=now())
            else:
                value["attempts"] += 1
                try:
                    item = read_json(Path(value["path"]))
                    receipt = self.transport(self.config, item, Path(value["path"]))
                    value.update(status="queued", queued_at=now(), receipt=receipt)
                    value.pop("last_error", None)
                except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
                    value.update(last_error=str(error), retry_after=time.time() + min(600, 30 * 2 ** min(value["attempts"], 5)))
            atomic_json(self.state_path, self.state)
            if value["status"] == "pending":
                break  # Preserve report order if transport is unavailable.
        counts = {name: sum(v["status"] == name for v in self.state["reports"].values()) for name in ("queued", "pending", "acknowledged")}
        done = progress.get("status") == "complete" and any(x["kind"] == "complete" for x in found) and not counts["pending"]
        atomic_json(self.root / "status.json", {"updated": now(), "pid": os.getpid(), "status": "complete" if done else "watching",
                    "thread_id": self.config["thread_id"], "poll_seconds": self.config.get("poll_seconds", 30),
                    "training_status": progress.get("status"), "training_stage": progress.get("stage"),
                    "training_updates": progress.get("total_updates"), "reports": counts})
        return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ack")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    config = read_json(args.config)
    if not config:
        raise SystemExit("Missing or invalid reporting configuration.")
    root = Path(config["run_dir"]).resolve() / "reporting"
    if root.drive.upper() != "C:" or args.config.resolve() != root / "config.json":
        raise SystemExit("Reporting configuration must be in the C-drive run's reporting directory.")
    if args.ack:
        if not re.fullmatch(r"[a-f0-9]{24}", args.ack) or not (root / "reports" / (args.ack + ".json")).is_file():
            raise SystemExit("Unknown report ID.")
        atomic_json(root / "acknowledged" / (args.ack + ".json"), {"id": args.ack, "at": now()})
        print("Report acknowledged: " + args.ack)
        return
    import msvcrt
    with (root / "watch.lock").open("a+b") as lock:
        if lock.tell() == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            raise SystemExit("A result watcher is already running.")
        atomic_json(root / "owner.json", {"pid": os.getpid(), "started": now()})
        watcher = Watcher(config)
        while not (root / "stop.request").exists():
            try:
                if watcher.scan() or args.once:
                    return
            except Exception as error:
                atomic_json(root / "status.json", {"status": "watcher_error_retrying", "pid": os.getpid(), "updated": now(), "error": repr(error)})
                print(f"{now()} {error!r}", flush=True)
            time.sleep(config.get("poll_seconds", 30))
        atomic_json(root / "status.json", {"status": "stopped", "pid": os.getpid(), "updated": now()})


if __name__ == "__main__":
    main()
