#!/usr/bin/env python3

"""Generate scene-specific Habitat experiment scripts through scene_script.py."""

import argparse
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from config import CFG


SCENES = {
    "808": ("00808-y9hTuugGdiq", "y9hTuugGdiq"),
    "810": ("00810-CrMo8WxCyVb", "CrMo8WxCyVb"),
    "813": ("00813-svBbv1Pavdk", "svBbv1Pavdk"),
    "814": ("00814-p53SfW6mjZe", "p53SfW6mjZe"),
    "815": ("00815-h1zeeAwLh9Z", "h1zeeAwLh9Z"),
    "820": ("00820-mL8ThkuaVTM", "mL8ThkuaVTM"),
    "821": ("00821-eF36g7L6Z9M", "eF36g7L6Z9M"),
    "824": ("00824-Dd4bFSTQ8gi", "Dd4bFSTQ8gi"),
    "827": ("00827-BAbdmeyTvMZ", "BAbdmeyTvMZ"),
    "829": ("00829-QaLdnwvtxbs", "QaLdnwvtxbs"),
}

SCENARIO_NAMES = ("relocate_all", "relocate_remove", "two_phase")
MAX_RAW_STEPS = 64
MAX_COMPILED_STEPS = 127


def profile_requirements(object_count, scenario):
    """Return hard action constraints for the first, prompt-writing LLM call."""
    if scenario == "relocate_remove":
        remove_count = max(1, object_count // 3)
        return (
            f"Create exactly {object_count} different objects. Wait exactly 10 seconds; "
            "move every object exactly once to a new destination; wait exactly 5 "
            f"seconds; then remove exactly {remove_count} of the moved objects."
        )
    if scenario == "two_phase":
        if object_count == 1:
            return (
                "Create exactly 1 object. Wait exactly 10 seconds; move it exactly "
                "once to a new destination; do not remove it."
            )
        first_group = object_count // 2
        return (
            f"Create exactly {object_count} different objects. Wait exactly 5 seconds; "
            f"move {first_group} objects exactly once; wait exactly 5 seconds; then "
            "move every remaining object exactly once. Every move must use a new "
            "destination. Do not remove anything."
        )
    return (
        f"Create exactly {object_count} different objects. Wait exactly 10 seconds; "
        "then move every object exactly once to a new destination. Do not remove "
        "anything."
    )


def load_openrouter_config():
    config_path = Path(__file__).with_name("openrouter.txt")
    try:
        lines = config_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return {}
    settings = {}
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, value = line.partition("=")
        if not separator:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        settings[key.strip()] = value
    return settings


def config_value(name, config, default=""):
    return os.environ.get(name, config.get(name, default)).strip()


def openrouter_json_request(request, context="richiesta OpenRouter"):
    """Execute an OpenRouter request with bounded retry on transient failures."""
    try:
        max_attempts = int(os.environ.get("OPENROUTER_HTTP_RETRIES", "5"))
    except ValueError:
        max_attempts = 5
    try:
        base_delay = float(os.environ.get("OPENROUTER_RETRY_BASE_SECONDS", "3"))
    except ValueError:
        base_delay = 3.0
    max_attempts = min(10, max(1, max_attempts))
    base_delay = min(30.0, max(0.1, base_delay))
    retryable_statuses = {408, 409, 429, 500, 502, 503, 504}
    for attempt in range(1, max_attempts + 1):
        try:
            with urlopen(request, timeout=300) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")[:1000]
            if exc.code not in retryable_statuses or attempt == max_attempts:
                raise RuntimeError(
                    f"{context} fallita: HTTP {exc.code}: {details}"
                ) from exc
            retry_after = None
            if exc.headers is not None:
                try:
                    retry_after = float(exc.headers.get("Retry-After", ""))
                except (TypeError, ValueError):
                    retry_after = None
            delay = min(
                60.0,
                retry_after if retry_after is not None
                else base_delay * (2 ** (attempt - 1)),
            )
            print(
                f"{context}: HTTP {exc.code}, nuovo tentativo "
                f"{attempt + 1}/{max_attempts} tra {delay:g}s.",
                flush=True,
            )
            time.sleep(delay)
        except OSError as exc:
            raise RuntimeError(f"{context} fallita: {exc}") from exc
    raise AssertionError("ciclo retry OpenRouter terminato senza risultato")


def decode_prompt_response(result):
    content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
    text = str(content).strip()
    if text.startswith("```") and text.endswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError("OpenRouter non ha restituito un prompt JSON valido") from exc
    title = str(data.get("title", "")).strip()
    prompt = str(data.get("prompt", "")).strip()
    if not title or not prompt:
        raise RuntimeError("OpenRouter ha restituito title o prompt vuoto")
    return title, prompt


def available_template_names(objects_dir):
    suffix = ".object_config.json"
    return sorted(
        path.name[:-len(suffix)] for path in objects_dir.glob(f"*{suffix}")
    )


def mentioned_templates(prompt, templates):
    normalized = str(prompt).lower()
    return sorted({name for name in templates if name.lower() in normalized})


def failed_placement_template(log_text, templates):
    """Extract a template that made placement impossible from compiler output."""
    patterns = (
        r"nessun placement valido per\s+([^\s]+)",
        r"nessuna superficie .*?template\s+['\"]([^'\"]+)['\"]",
    )
    available = set(templates)
    for pattern in patterns:
        match = re.search(pattern, str(log_text), re.IGNORECASE)
        if match:
            candidate = match.group(1).strip(".,;:'\"")
            if candidate in available:
                return candidate
    return None


def run_compiler(command, environment):
    """Run scene_script while preserving live diagnostics and a parseable log."""
    process = subprocess.Popen(
        command, env=environment, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    lines = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    return process.wait(), "".join(lines)


def generate_unique_prompt(
    object_count, scenario, scene_id, duration_minutes, guidance,
    previous_prompts, templates, attempt_limit=3,
):
    """Ask OpenRouter for one unique natural-language experiment request."""
    config = load_openrouter_config()
    api_key = config_value("OPENROUTER_API_KEY", config)
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY non impostata: serve per generare i prompt del batch"
        )
    base_url = config_value(
        "OPENROUTER_URL", config, "https://openrouter.ai/api/v1"
    ).rstrip("/")
    endpoint = (
        base_url if base_url.endswith("/chat/completions")
        else base_url + "/chat/completions"
    )
    model = config_value(
        "OPENROUTER_PROMPT_MODEL", config,
        config_value(
            "OPENROUTER_MODEL", config, "qwen/qwen3-30b-a3b-instruct-2507"
        ),
    )
    schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string", "minLength": 3},
            "prompt": {"type": "string", "minLength": 40},
        },
        "required": ["title", "prompt"],
        "additionalProperties": False,
    }
    normalized_previous = {" ".join(item.lower().split()) for item in previous_prompts}
    previous_excerpt = previous_prompts[-9:]
    requirements = profile_requirements(object_count, scenario)
    for attempt in range(1, attempt_limit + 1):
        instructions = (
            "Write one creative English-language user request for a Habitat-Sim "
            "experiment. The request will be passed verbatim to a separate geometry-aware "
            "compiler. Make its theme, object mix, wording, and manipulation story "
            "materially different from all previous requests.\n"
            f"Scene identifier (for differentiation only): {scene_id}.\n"
            f"Approximate final runtime (handled later by the batch): {duration_minutes:g} minutes.\n"
            f"Mandatory action constraints: {requirements}\n"
            f"Additional user guidance: {guidance or 'Use a plausible everyday theme.'}\n"
            "The prompt must explicitly repeat the exact numeric object count and every "
            "mandatory action constraint. Select exactly that many DISTINCT identifiers "
            "from Available object templates and copy each identifier verbatim into the "
            "prompt. Give each selected object a unique logical name. Never invent an "
            "object type that is absent from this list. Do not name coordinates, placement "
            "IDs, room categories, furniture, or support categories: only the downstream "
            "compiler knows what this scene actually contains. Do not add actions or waits "
            "beyond the mandatory ones. "
            "Do not mention these instructions, the scene identifier, or runtime estimation.\n"
            "Available object templates:\n" + json.dumps(templates, ensure_ascii=False) + "\n"
            "Previous requests to avoid:\n" + json.dumps(previous_excerpt, ensure_ascii=False)
        )
        if attempt > 1:
            instructions += "\nThe prior candidate was duplicate or invalid; make this one clearly different."
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": instructions}],
            "stream": False,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "experiment_prompt",
                    "strict": True,
                    "schema": schema,
                },
            },
            "temperature": 0.9,
            "max_tokens": 700,
            "provider": {"require_parameters": True},
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        site_url = config_value("OPENROUTER_SITE_URL", config)
        site_name = config_value("OPENROUTER_SITE_NAME", config, "lost3dsg")
        if site_url:
            headers["HTTP-Referer"] = site_url
        if site_name:
            headers["X-OpenRouter-Title"] = site_name
        request = Request(
            endpoint, data=json.dumps(payload).encode("utf-8"),
            headers=headers, method="POST",
        )
        result = openrouter_json_request(
            request, context="generazione prompt OpenRouter"
        )
        title, prompt = decode_prompt_response(result)
        normalized = " ".join(prompt.lower().split())
        count_is_explicit = bool(re.search(rf"\b{object_count}\b", prompt))
        selected_templates = mentioned_templates(prompt, templates)
        if (
            normalized not in normalized_previous
            and count_is_explicit
            and len(selected_templates) == object_count
        ):
            return title, prompt
    raise RuntimeError(
        f"OpenRouter non ha prodotto un prompt unico valido per la scena {scene_id}"
    )


def projected_counts(object_count, scenario):
    """Return conservative raw/compiled/action counts for a generated scenario."""
    if scenario == "relocate_remove":
        removals = max(1, object_count // 3)
        raw = 2 * object_count + removals + 2
        compiled = raw + max(0, object_count - 1) * 2
        return raw, compiled, 2 * object_count + removals, 15.0
    if scenario == "two_phase" and object_count > 1:
        raw = 2 * object_count + 2
        compiled = raw + max(0, object_count - 1) + max(0, object_count - 2)
        return raw, compiled, 2 * object_count, 10.0
    raw = 2 * object_count + 1
    compiled = raw + max(0, object_count - 1) * 2
    return raw, compiled, 2 * object_count, 10.0


def validate_job_budget(object_count, scenario, target_seconds, action_overhead):
    raw, compiled, action_count, explicit_wait = projected_counts(
        object_count, scenario
    )
    if raw > MAX_RAW_STEPS or compiled > MAX_COMPILED_STEPS:
        raise ValueError(
            f"{scenario} con {object_count} oggetti richiede circa {raw} step "
            f"grezzi/{compiled} compilati; limiti {MAX_RAW_STEPS}/{MAX_COMPILED_STEPS}"
        )
    minimum_seconds = explicit_wait + action_count * float(action_overhead)
    if minimum_seconds > target_seconds:
        raise ValueError(
            f"{scenario} con {object_count} oggetti richiede almeno "
            f"{minimum_seconds:.1f}s secondo l'overhead configurato; target "
            f"{target_seconds:.1f}s"
        )


def selected_scenes(scene_count, scene_ids=None, randomize=False, seed=0):
    if scene_ids:
        ids = [item.strip().lstrip("0") or "0" for item in scene_ids.split(",")]
        unknown = [item for item in ids if item not in SCENES]
        if unknown:
            raise ValueError("scene sconosciute: " + ", ".join(unknown))
        if len(set(ids)) != len(ids):
            raise ValueError("--scene-ids contiene duplicati")
        if scene_count is not None and scene_count != len(ids):
            raise ValueError("--scene-count deve coincidere con il numero di --scene-ids")
        return ids
    count = len(SCENES) if scene_count is None else int(scene_count)
    if not 1 <= count <= len(SCENES):
        raise ValueError(f"--scene-count deve essere tra 1 e {len(SCENES)}")
    ids = list(SCENES)
    if randomize:
        ids = random.Random(seed).sample(ids, count)
    return ids[:count]


def tune_duration(data, target_seconds, action_overhead):
    """Tune generated settling waits to an approximate wall-clock duration."""
    steps = list(data.get("steps", []))
    action_count = sum(
        step.get("action") in {"spawn", "move", "remove"} for step in steps
    )
    explicit_wait = sum(
        float(step.get("seconds", 0.0))
        for step in steps
        if step.get("action") == "wait" and step.get("reason") != "settle"
    )
    settle_steps = [
        step for step in steps
        if step.get("action") == "wait" and step.get("reason") == "settle"
    ]
    fixed_seconds = explicit_wait + action_count * float(action_overhead)
    if fixed_seconds > target_seconds:
        raise ValueError(
            f"durata richiesta troppo breve: overhead stimato {fixed_seconds:.1f}s"
        )
    remaining = float(target_seconds) - fixed_seconds
    settle_seconds = remaining / len(settle_steps) if settle_steps else 0.0
    settle_seconds = min(60.0, max(0.0, settle_seconds))
    for step in settle_steps:
        step["seconds"] = round(settle_seconds, 3)
    estimated = fixed_seconds + settle_seconds * len(settle_steps)

    # With very few actions, 60-second settling waits may not reach the target.
    # Insert explicit padding before the first move so objects remain observable.
    deficit = max(0.0, float(target_seconds) - estimated)
    padding = []
    while deficit > 1e-6:
        seconds = min(60.0, deficit)
        padding.append({
            "action": "wait", "seconds": round(seconds, 3),
            "reason": "duration_padding",
        })
        deficit -= seconds
    if padding:
        insertion = next(
            (index for index, step in enumerate(steps) if step.get("action") == "move"),
            len(steps),
        )
        steps[insertion:insertion] = padding
    if len(steps) > 127:
        raise ValueError("la calibrazione della durata supera il limite di 127 step")

    data["steps"] = steps
    data["target_duration_seconds"] = float(target_seconds)
    data["estimated_duration_seconds"] = round(
        fixed_seconds + settle_seconds * len(settle_steps)
        + sum(float(step["seconds"]) for step in padding),
        3,
    )
    data["duration_model"] = {
        "action_count": action_count,
        "seconds_per_action": float(action_overhead),
        "explicit_wait_seconds": explicit_wait,
        "settle_wait_count": len(settle_steps),
        "settle_seconds": round(settle_seconds, 3),
        "padding_wait_count": len(padding),
    }
    return data


def scene_paths(habitat_root, scene_id):
    directory, asset = SCENES[scene_id]
    scene_dir = habitat_root / "hm3d-val-habitat-v0.2" / directory
    return scene_dir / f"{asset}.basis.glb", scene_dir / f"{asset}.basis.navmesh"


def build_jobs(
    scene_ids, scripts_per_scene, habitat_root, output_dir,
    batch_name, object_count,
):
    """Expand each selected scene into independently generated script jobs."""
    jobs = []
    variant_index = 0
    digits = max(2, len(str(scripts_per_scene)))
    for scene_id in scene_ids:
        scene, navmesh = scene_paths(habitat_root, scene_id)
        for script_index in range(1, scripts_per_scene + 1):
            scenario = SCENARIO_NAMES[variant_index % len(SCENARIO_NAMES)]
            if scripts_per_scene == 1:
                filename = f"{batch_name}_scene_{scene_id}.json"
                output = output_dir / filename
            else:
                filename = f"script_{script_index:0{digits}d}.json"
                output = (
                    output_dir / batch_name / f"scene_{scene_id}" / filename
                )
            jobs.append({
                "scene_id": scene_id,
                "script_index": script_index,
                "scripts_per_scene": scripts_per_scene,
                "scene": scene,
                "navmesh": navmesh,
                "scenario": scenario,
                "prompt_requirements": profile_requirements(
                    object_count, scenario
                ),
                "output": output,
            })
            variant_index += 1
    return jobs


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Genera con OpenRouter un prompt unico per scena e lo compila con "
            "scene_script.py."
        )
    )
    parser.add_argument(
        "--objects", type=int, required=True,
        help="Numero di oggetti da creare e manipolare in ogni scena (1-30).",
    )
    parser.add_argument(
        "--object-scale", type=float, default=None,
        help=(
            "Scala uniforme degli oggetti inoltrata a scene_script.py; "
            "per esempio 0.75 o 1.2."
        ),
    )
    parser.add_argument(
        "--scene-count", type=int, default=None,
        help="Numero di scene da usare; senza --scene-ids prende le prime N.",
    )
    parser.add_argument(
        "--scripts-per-scene", type=int, default=1,
        help="Numero di JSON indipendenti da generare per ciascuna scena (default: 1).",
    )
    parser.add_argument(
        "--scene-ids", default="",
        help="ID espliciti separati da virgole, per esempio 808,815,829.",
    )
    parser.add_argument(
        "--duration-minutes", type=float, required=True,
        help="Durata stimata di ciascun JSON generato, in minuti.",
    )
    parser.add_argument(
        "--action-overhead", type=float, default=4.0,
        help="Secondi stimati per spawn/move/remove, inclusa la cattura (default: 4).",
    )
    parser.add_argument(
        "--random-scenes", action="store_true",
        help="Campiona le scene invece di prendere le prime N.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-name", default="experiment")
    parser.add_argument("--output-dir", type=Path, default=Path("scripts/generated"))
    parser.add_argument(
        "--prompt-guidance", default="",
        help=(
            "Caratteristiche aggiuntive per i prompt, per esempio temi domestici "
            "diversi e manipolazioni realistiche."
        ),
    )
    parser.add_argument(
        "--prompt-retries", type=int, default=3,
        help="Tentativi OpenRouter se il prompt è duplicato o non valido (default: 3).",
    )
    parser.add_argument(
        "--compile-retries", type=int, default=3,
        help=(
            "Nuovi prompt per scena se un template non trova placement "
            "sufficienti (default: 3)."
        ),
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Salta i JSON già generati e riparte dalla prima scena mancante.",
    )
    parser.add_argument(
        "--habitat-root", type=Path, default=Path(CFG["habitat"]["dataset_root"]),
        help="radice dei dataset Habitat (default: HABITAT_DATASETS_DIR)",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    if not 1 <= args.objects <= 30:
        raise ValueError("--objects deve essere tra 1 e 30")
    if args.object_scale is not None and (
        not math.isfinite(args.object_scale) or args.object_scale <= 0
    ):
        raise ValueError("--object-scale deve essere un numero positivo")
    if not math.isfinite(args.duration_minutes) or args.duration_minutes <= 0:
        raise ValueError("--duration-minutes deve essere positivo")
    if not math.isfinite(args.action_overhead) or args.action_overhead < 0:
        raise ValueError("--action-overhead deve essere non negativo")
    if not 1 <= args.prompt_retries <= 10:
        raise ValueError("--prompt-retries deve essere tra 1 e 10")
    if not 1 <= args.compile_retries <= 10:
        raise ValueError("--compile-retries deve essere tra 1 e 10")
    if not 1 <= args.scripts_per_scene <= 50:
        raise ValueError("--scripts-per-scene deve essere tra 1 e 50")
    scene_ids = selected_scenes(
        args.scene_count, args.scene_ids, args.random_scenes, args.seed
    )
    dataset = (
        args.habitat_root / "hm3d-val-semantic-configs-v0.2"
        / "hm3d_annotated_basis.scene_dataset_config.json"
    )
    objects_dir = args.habitat_root / "habitat_objects" / "configs"
    templates = available_template_names(objects_dir)
    if objects_dir.is_dir() and args.objects > len(templates):
        raise ValueError(
            f"--objects={args.objects}, ma sono disponibili solo {len(templates)} template"
        )
    target_seconds = args.duration_minutes * 60.0
    jobs = build_jobs(
        scene_ids, args.scripts_per_scene, args.habitat_root,
        args.output_dir, args.batch_name, args.objects,
    )
    for job in jobs:
        job["requested_object_scale"] = args.object_scale

    # Validate every variant before making the first model request, avoiding a
    # partially generated batch and unnecessary API usage.
    for job in jobs:
        validate_job_budget(
            args.objects, job["scenario"], target_seconds, args.action_overhead
        )

    if args.dry_run:
        print(json.dumps([
            {key: str(value) if isinstance(value, Path) else value for key, value in job.items()}
            for job in jobs
        ], indent=2, ensure_ascii=False))
        return 0

    missing = [
        str(path) for job in jobs for path in (job["scene"], job["navmesh"])
        if not path.is_file()
    ]
    if not dataset.is_file():
        missing.append(str(dataset))
    if not objects_dir.is_dir():
        missing.append(str(objects_dir))
    if missing:
        raise FileNotFoundError("asset mancanti:\n" + "\n".join(missing))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Batch: {len(scene_ids)} scene x {args.scripts_per_scene} script = "
        f"{len(jobs)} JSON da generare.",
        flush=True,
    )
    summary = []
    generated_prompts = []
    for job in jobs:
        job_label = (
            f"scene {job['scene_id']} script "
            f"{job['script_index']}/{job['scripts_per_scene']}"
        )
        if args.resume and job["output"].is_file():
            existing = json.loads(job["output"].read_text(encoding="utf-8"))
            previous_prompt = str(existing.get("generation_prompt", "")).strip()
            existing_templates = mentioned_templates(previous_prompt, templates)
            existing_scale = existing.get("object_scale")
            scale_matches = (
                args.object_scale is None
                or (
                    isinstance(existing_scale, (int, float))
                    and math.isclose(
                        float(existing_scale), args.object_scale,
                        rel_tol=1e-9, abs_tol=1e-9,
                    )
                )
            )
            if (
                previous_prompt
                and len(existing_templates) == args.objects
                and scale_matches
            ):
                generated_prompts.append(previous_prompt)
                print(
                    f"Skipping existing {job_label} -> {job['output']}",
                    flush=True,
                )
                summary.append({
                    "scene_id": job["scene_id"], "scenario": job["scenario"],
                    "script_index": job["script_index"],
                    "prompt_title": existing.get("generation_prompt_title", ""),
                    "output": str(job["output"]),
                    "estimated_duration_seconds": existing.get(
                        "estimated_duration_seconds"
                    ),
                    "skipped": True,
                })
                continue
            reasons = []
            if len(existing_templates) != args.objects:
                reasons.append(
                    f"il vecchio prompt cita {len(existing_templates)}/"
                    f"{args.objects} template reali"
                )
            if not scale_matches:
                reasons.append(
                    f"scala esistente {existing_scale!r}, richiesta {args.object_scale}"
                )
            print(
                f"Regenerating {job_label}: " + "; ".join(reasons) + ".",
                flush=True,
            )
        environment = dict(os.environ)
        # Emit settle markers; tune_duration replaces their duration afterwards.
        environment["HABITAT_SETTLE_SECONDS"] = "1.0"
        excluded_templates = set()
        attempted_prompts = []
        data = None
        prompt = ""
        prompt_title = ""
        last_returncode = 1
        for compile_attempt in range(1, args.compile_retries + 1):
            allowed_templates = [
                item for item in templates if item not in excluded_templates
            ]
            if len(allowed_templates) < args.objects:
                raise RuntimeError(
                    f"template utilizzabili insufficienti per la scena {job['scene_id']}"
                )
            print(
                f"Generating unique prompt for {job_label} "
                f"({job['scenario']}), compile attempt "
                f"{compile_attempt}/{args.compile_retries}",
                flush=True,
            )
            prompt_title, prompt = generate_unique_prompt(
                args.objects, job["scenario"],
                f"{job['scene_id']}/script-{job['script_index']}",
                args.duration_minutes, args.prompt_guidance,
                generated_prompts + attempted_prompts, allowed_templates,
                attempt_limit=args.prompt_retries,
            )
            attempted_prompts.append(prompt)
            print(f"Prompt [{prompt_title}]: {prompt}", flush=True)
            print(f"Compiling {job_label} -> {job['output']}", flush=True)
            temporary = tempfile.NamedTemporaryFile(
                prefix=(
                    f"{args.batch_name}_{job['scene_id']}_"
                    f"{job['script_index']}_"
                ),
                suffix=".json", delete=False,
            )
            temporary_output = Path(temporary.name)
            temporary.close()
            command = [
                sys.executable, str(Path(__file__).with_name("scene_script.py")),
                "--scene", str(job["scene"]),
                "--scene-dataset", str(dataset),
                "--navmesh", str(job["navmesh"]),
                "--objects-dir", str(objects_dir),
                "--request", prompt,
                "--output", str(temporary_output),
            ]
            if args.object_scale is not None:
                command.extend(["--object-scale", str(args.object_scale)])
            try:
                last_returncode, compiler_log = run_compiler(command, environment)
                if last_returncode == 0:
                    data = json.loads(temporary_output.read_text(encoding="utf-8"))
                    break
            finally:
                temporary_output.unlink(missing_ok=True)
            failed_template = failed_placement_template(compiler_log, templates)
            if failed_template:
                excluded_templates.add(failed_template)
                print(
                    f"Retry {job_label}: escludo {failed_template} "
                    "e genero un nuovo prompt.",
                    flush=True,
                )
            else:
                print(
                    f"Retry {job_label}: genero un nuovo prompt dopo "
                    "il fallimento della compilazione.",
                    flush=True,
                )
        if data is None:
            excluded = ", ".join(sorted(excluded_templates)) or "nessuno"
            raise RuntimeError(
                f"generazione fallita per {job_label} dopo "
                f"{args.compile_retries} prompt (ultimo exit {last_returncode}; "
                f"template esclusi: {excluded})"
            )
        generated_prompts.append(prompt)
        tune_duration(data, target_seconds, args.action_overhead)
        data["scene_id"] = job["scene_id"]
        data["script_index"] = job["script_index"]
        data["scripts_per_scene"] = job["scripts_per_scene"]
        data["requested_object_scale"] = args.object_scale
        data["scene"] = str(job["scene"])
        data["navmesh"] = str(job["navmesh"])
        data["scenario"] = job["scenario"]
        data["generation_prompt_title"] = prompt_title
        data["generation_prompt"] = prompt
        data["prompt_guidance"] = args.prompt_guidance
        data["compile_attempts"] = len(attempted_prompts)
        data["excluded_templates"] = sorted(excluded_templates)
        job["output"].parent.mkdir(parents=True, exist_ok=True)
        job["output"].write_text(json.dumps(data, indent=2), encoding="utf-8")
        summary.append({
            "scene_id": job["scene_id"], "scenario": job["scenario"],
            "script_index": job["script_index"],
            "prompt_title": prompt_title,
            "output": str(job["output"]),
            "estimated_duration_seconds": data["estimated_duration_seconds"],
        })
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"Errore batch: {exc}")
        raise SystemExit(1) from None
