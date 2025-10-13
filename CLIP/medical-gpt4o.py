# tak_two_agent.py
import os
import re
import json
import time
import base64
import random
import logging
import traceback
from io import BytesIO
from copy import deepcopy
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from PIL import Image

# ------------------------ Utils ------------------------

def setup_logging(level=logging.INFO):
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s",
        level=level
    )

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def encode_image_to_b64(image_path: str, side: int = 512) -> str:
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    if w >= h:
        new_w = side
        new_h = int(side * h / w)
    else:
        new_h = side
        new_w = int(side * w / h)
    img_resized = img.resize((new_w, new_h))
    buffer = BytesIO()
    img_resized.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def redact_image_urls_in_msgs(msgs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    msgs = deepcopy(msgs)
    for m in msgs:
        if isinstance(m.get("content"), list):
            for part in m["content"]:
                if part.get("type") == "image_url" and isinstance(part.get("image_url"), dict):
                    url = part["image_url"].get("url", "")
                    part["image_url"]["url"] = f"<base64 omitted, length={len(url)}>"
    return msgs

def safe_json_loads(s: str) -> Any:

    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.IGNORECASE|re.MULTILINE)

    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = s[start:end+1]
        try:
            return json.loads(candidate)
        except Exception:
            pass
    # Last resort: plain json.loads
    try:
        return json.loads(s)
    except Exception:
        return None

# ------------------------ HTTP / GPT client ------------------------

class GPTClient:
    def __init__(
        self,
        api_key: str = None,
        url: str = "https://api.ai-gaochao.cn/v1/chat/completions",
        model: str = "gpt-4o",
        temperature: float = 0.7,
        top_p: float = 0.8,
        presence_penalty: float = 1.0,
        n: int = 1,
        timeout: int = 120,
        max_retries: int = 3
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.url = url
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.presence_penalty = presence_penalty
        self.n = n
        self.timeout = timeout
        self.max_retries = max_retries

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def chat(self, messages: List[Dict[str, Any]], n: int = None) -> List[str]:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "presence_penalty": self.presence_penalty,
            "n": n or self.n,
        }

        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(self.url, headers=self.headers, json=payload, timeout=self.timeout)
                if resp.status_code != 200:
                    raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:400]}")
                data = resp.json()
                choices = data.get("choices", [])
                outs = []
                for c in choices:
                    outs.append(c["message"]["content"])
                return outs
            except Exception as e:
                last_err = e
                logging.warning(f"chat attempt {attempt+1}/{self.max_retries} failed: {e}")
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"chat failed after {self.max_retries} attempts: {last_err}")

# ------------------------ Agent 1: Text-only prior generator ------------------------

class PriorGeneratorAgent:

    SYSTEM_PROMPT = (
        "You are an expert medical anatomy writer for clinical/educational use. "
        "Strictly avoid sexualized or erotic language. "
        "Do not describe nudity; discuss organs only in non-sexual, clinical terms. "
        "No content about minors. No personally identifying info. "
        "When asked, respond with STRICT JSON with keys 'positions' and 'shape', each an array of 5-10 short strings. "
        "Use precise, compact statements. Assume healthy adult anatomy unless noted. "
        "Avoid view-specific descriptors. Use ONLY the provided organ category list."
    )


    TEMPLATE_POS = "Describe the relative positional relationship of [CLS] with other organs."
    TEMPLATE_SHAPE = "Describe the shape and structure of [CLS]."

    def __init__(self, client: GPTClient, categories: List[str]):
        self.client = client
        self.categories = categories

    def _prompt_positions(self, cls: str) -> List[Dict[str, Any]]:
        cat_str = ", ".join(self.categories)
        user_text = (
            "Context: Clinical/educational anatomy writing. Non-sexual, non-graphic, de-identified.\n"
            f"Categories to reference ONLY: [{cat_str}].\n\n"
            f"{self.TEMPLATE_POS.replace('[CLS]', cls)}\n\n"
            f"Return STRICT JSON with keys 'positions' and 'shape'. For now, fill 'positions' only; put an empty array [] for 'shape'."
        )
        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ]

    def _prompt_shape(self, cls: str) -> List[Dict[str, Any]]:
        cat_str = ", ".join(self.categories)
        # Use exact template for shape
        user_text = (
            f"Categories to reference ONLY: [{cat_str}].\n\n"
            f"{self.TEMPLATE_SHAPE.replace('[CLS]', cls)}\n\n"
            f"Return STRICT JSON with keys 'positions' and 'shape'. For now, fill 'shape' only; put an empty array [] for 'positions'."
        )
        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ]

    def generate_for_class(self, cls: str) -> Dict[str, List[str]]:
        pos_out = self.client.chat(self._prompt_positions(cls))[0]
        shp_out = self.client.chat(self._prompt_shape(cls))[0]
        jp = safe_json_loads(pos_out) or {"positions": [], "shape": []}
        js = safe_json_loads(shp_out) or {"positions": [], "shape": []}
        positions = jp.get("positions", [])
        shape = js.get("shape", [])
        positions = list(dict.fromkeys([s.strip() for s in positions if isinstance(s, str) and s.strip()]))
        shape = list(dict.fromkeys([s.strip() for s in shape if isinstance(s, str) and s.strip()]))
        return {"positions": positions, "shape": shape}

    def generate_all(self, max_workers: int = 4) -> Dict[str, Dict[str, List[str]]]:
        out = {}
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(self.generate_for_class, c): c for c in self.categories}
            for fut in as_completed(futs):
                cls = futs[fut]
                try:
                    out[cls] = fut.result()
                except Exception as e:
                    logging.error(f"Agent1 failed for class '{cls}': {e}")
                    out[cls] = {"positions": [], "shape": []}
        return out

# ------------------------ Agent 2: Multimodal validator (6 views per case, 2 cases -> 12 checks) ------------------------

class MultimodalValidatorAgentSixViews:
    SYSTEM_PROMPT = (
        "You are a meticulous multimodal validator working on de-identified medical renders for clinical research. "
        "Use only non-sexual, clinical language. No minors. No personal data. "
        "You will be given a textual claim about an organ and 12 rendered images (canonical viewpoints). "
        "You will be given a textual claim about an organ and a set of rendered images (each image is one canonical viewpoint of the same or different cases). "
        "Viewpoints are: Front, Back, Left, Right, Up, Down (in that order per case). "
        "Each organ in the images is encoded by a unique RGB color; an RGB->class table will be provided. "
        "For each image, judge whether the image **individually** shows evidence that the claim 'matches', is 'unclear', or 'does not match'. "
        "Return STRICT JSON with keys: "
        "{"
        "\"per_image\": [ {\"image\": \"<filename>\", \"viewpoint\": \"Front|Back|Left|Right|Up|Down\", \"verdict\": \"matches|unclear|does not match\", \"reason\": \"<short>\"}, ... (12 entries) ], "
        "\"summary\": {\"does_not_match_count\": <int>, \"matches_count\": <int>, \"unclear_count\": <int>}, "
        "\"overall\": \"matches|unclear|does not match\" "
        "}. "
        "Do NOT use external knowledge beyond the image content and provided RGB->class mapping and viewpoint info."
    )

    VIEWPOINT_ORDER = ["Front", "Back", "Left", "Right", "Up", "Down"]

    def __init__(
        self,
        client: GPTClient,
        image_dir: str,
        metadata_cases: Dict[str, Dict[str, Any]],
        samples_per_class: int = 2,  # per manuscript: randomly select two cases
        max_workers: int = 6,
        views_per_case: int = 6,
        require_exact_six: bool = True,
        preset_r: List[int] = None,
        preset_g: List[int] = None,
        preset_b: List[int] = None
    ):

        self.client = client
        self.image_dir = image_dir
        self.meta = metadata_cases
        self.samples_per_class = samples_per_class
        self.max_workers = max_workers
        self.views_per_case = views_per_case
        self.require_exact_six = require_exact_six


        self.preset_r = preset_r
        self.preset_g = preset_g
        self.preset_b = preset_b

        self.by_class = self._invert_index(self.meta)

    def _invert_index(self, meta: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
        idx: Dict[str, List[str]] = {}
        for case_id, obj in meta.items():
            classes = obj.get("classes", [])
            if not isinstance(classes, list):
                continue
            imgs = obj.get("images", [])
            if not isinstance(imgs, list) or len(imgs) == 0:
                continue
            if self.require_exact_six and len(imgs) != self.views_per_case:
                continue
            for c in classes:
                idx.setdefault(c, []).append(case_id)
        return idx

    def _build_rgb_mapping(self, categories: List[str]) -> Dict[str, Tuple[int,int,int]]:
        r = [245, 243, 237, 141, 153, 216, 222, 222, 254, 124, 56, 134, 173, 168, 247]
        g = [245, 217, 157, 233, 242, 191, 222, 90, 227, 190, 65, 220, 159, 210, 231]
        b = [220, 172, 111, 141, 19, 216, 74, 107, 224, 230, 239, 134, 246, 224, 106]

        mapping = {}
        for i, c in enumerate(categories):
            mapping[c] = (r[i], g[i], b[i])
        return mapping

    def _build_validation_message(self, images_b64: List[Tuple[str, str]], claim: str, cls: str, rgb_map: Dict[str, Tuple[int,int,int]]) -> List[Dict[str, Any]]:
        rgb_lines = [f"{k}: {v[0]},{v[1]},{v[2]}" for k, v in rgb_map.items()]
        rgb_block = "\n".join(rgb_lines)

        text_block = (
            "Context: De-identified, synthetic medical segmentation renders for anatomy validation; "
            "an academic/clinical, non-sexual use case.\n"
            f"Class under review: [{cls}]\n"
            f"Claim: {claim}\n\n"
            f"Rendering strategy: Each case was rendered from 6 canonical viewpoints in this order: {', '.join(self.VIEWPOINT_ORDER)}. "
            f"For each case the 6 images are provided in that order. Two cases were sampled randomly, producing 12 images total.\n\n"
            f"RGB -> class mapping (each organ is rendered with the listed RGB value):\n{rgb_block}\n\n"
            f"Instruction: For each image (provided below in order), judge whether the claim 'matches', is 'unclear', or 'does not match' based ONLY on visible evidence in that image and the RGB->class mapping. "
            f"Return STRICT JSON as described in the system prompt. Be concise in reasons.\n"
        )


        user_content_list = [{"type": "text", "text": text_block}]

        for fname, b64 in images_b64:
            user_content_list.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}",
                    "detail": "low"
                }
            })
        
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_content_list} # The content is now a list of text and images
        ]
        
        return messages

    def _load_case_images_b64(self, case_id: str, images: List[str]) -> List[Tuple[str, str]]:
        imgs = list(images)
        if len(imgs) < self.views_per_case:
            while len(imgs) < self.views_per_case:
                imgs.append(imgs[-1])
        elif len(imgs) > self.views_per_case:
            imgs = imgs[:self.views_per_case]

        out = []
        for fn in imgs:
            path = os.path.join(self.image_dir, fn)
            b64 = encode_image_to_b64(path)
            out.append((fn, b64))
        return out

    def _validate_two_cases(self, case_ids: List[str], claim: str, cls: str, rgb_map: Dict[str, Tuple[int,int,int]]) -> Dict[str, Any]:

        try:
            images_b64 = []
            case_infos = []
            for cid in case_ids:
                obj = self.meta[cid]
                imgs = obj.get("images", [])
                case_img_b64 = self._load_case_images_b64(cid, imgs)  # list of 6 tuples
                images_b64.extend(case_img_b64)
                case_infos.append({"case_id": cid, "images": [p for p, _ in case_img_b64]})

            msgs = self._build_validation_message(images_b64, claim, cls, rgb_map)
            out = self.client.chat(msgs)[0]
            js = safe_json_loads(out) or {}
            # Normalize per_image entries to exactly 12
            per_image = js.get("per_image", [])
            if not isinstance(per_image, list) or len(per_image) == 0:
                names = [fn for fn, _ in images_b64]
                per_image = []
                for i, name in enumerate(names):
                    vp = self.VIEWPOINT_ORDER[i % self.views_per_case]
                    per_image.append({"image": name, "viewpoint": vp, "verdict": "unclear", "reason": ""})
            else:
                names = [fn for fn, _ in images_b64]
                fixed = []
                for i in range(self.views_per_case * len(case_ids)):
                    item = per_image[i] if i < len(per_image) else {}
                    vp = self.VIEWPOINT_ORDER[i % self.views_per_case]
                    fixed.append({
                        "image": item.get("image", names[i]),
                        "viewpoint": item.get("viewpoint", vp),
                        "verdict": str(item.get("verdict", "unclear")).strip().lower(),
                        "reason": item.get("reason", "")
                    })
                per_image = fixed

            for pi in per_image:
                v = pi.get("verdict", "").lower()
                if "match" in v and ("does" in v or "not" in v or "does not" in v):
                    pi["verdict"] = "does not match"
                elif v in {"consistent", "match", "matches", "yes"} or ("match" in v and not "does" in v):
                    pi["verdict"] = "matches"
                elif v in {"uncertain", "unclear", "unknown"}:
                    pi["verdict"] = "unclear"
                else:
                    pi["verdict"] = "unclear"

            counts = {"does_not_match": 0, "matches": 0, "unclear": 0}
            for pi in per_image:
                if pi["verdict"] == "does not match":
                    counts["does_not_match"] += 1
                elif pi["verdict"] == "matches":
                    counts["matches"] += 1
                else:
                    counts["unclear"] += 1

            overall = "unclear"
            denom = counts["matches"] + counts["does_not_match"]
            if denom > 0:
                overall = "matches" if counts["matches"] >= counts["does_not_match"] else "does not match"
            else:
                overall = "unclear"

            return {
                "case_ids": case_ids,
                "per_image": per_image,
                "summary": counts,
                "overall": overall
            }
        except Exception as e:
            return {
                "case_ids": case_ids,
                "per_image": [],
                "summary": {"does_not_match": 0, "matches": 0, "unclear": 0},
                "overall": "unclear",
                "error": str(e)
            }

    def validate_and_filter(
        self,
        priors: Dict[str, Dict[str, List[str]]],
        save_validation_dir: str = None,
        categories_list: List[str] = None
    ) -> Dict[str, Dict[str, List[str]]]:

        if save_validation_dir:
            ensure_dir(save_validation_dir)

        # Build RGB mapping used in prompts (deterministic)
        rgb_map = self._build_rgb_mapping(categories_list or [])

        filtered: Dict[str, Dict[str, List[str]]] = {}
        for cls, kv in priors.items():
            pos_claims = kv.get("positions", [])
            shp_claims = kv.get("shape", [])

            candidate_cases = self.by_class.get(cls, [])
            if not candidate_cases:
                logging.warning(f"No labeled cases with six views found for class '{cls}'. Skipping validation; passing priors through.")
                filtered[cls] = {"positions": pos_claims, "shape": shp_claims}
                continue

            # sample exactly two distinct cases (or fewer if not available)
            cases = list(candidate_cases)
            random.shuffle(cases)
            if len(cases) > self.samples_per_class:
                cases = cases[:self.samples_per_class]

            class_log = {"positions": [], "shape": []}
            keep_pos, keep_shp = [], []

            # Validate positions
            for claim in pos_claims:
                # If less than requested cases available, still use what's there.
                if len(cases) == 0:
                    logging.warning(f"No cases available to validate claim for class {cls}; keeping by default.")
                    keep_pos.append(claim)
                    continue

                # ensure we have two case ids: if only one available, duplicate it (but log)
                selected = list(cases)
                if len(selected) == 1:
                    logging.warning(f"Only one case available for class {cls}; duplicating to reach two-case validation.")
                    selected = [selected[0], selected[0]]

                # Run validation (single model call per-claim covering both cases)
                result = self._validate_two_cases(selected, claim, cls, rgb_map)
                # Save in class_log
                class_log["positions"].append({
                    "claim": claim,
                    "case_ids": selected,
                    "result": result
                })

                # Count how many 'does not match' among the 12 per-image judgments
                does_not_match_count = result.get("summary", {}).get("does_not_match", 0)
                # Filtering rule: if > 2 out of 12 are 'does not match', filter out
                if does_not_match_count > 2:
                    # filtered out -> do not keep
                    logging.info(f"Position claim filtered out for class {cls}; does_not_match_count={does_not_match_count}")
                else:
                    keep_pos.append(claim)

            # Validate shape
            for claim in shp_claims:
                if len(cases) == 0:
                    logging.warning(f"No cases available to validate claim for class {cls}; keeping by default.")
                    keep_shp.append(claim)
                    continue

                selected = list(cases)
                if len(selected) == 1:
                    selected = [selected[0], selected[0]]

                result = self._validate_two_cases(selected, claim, cls, rgb_map)
                class_log["shape"].append({
                    "claim": claim,
                    "case_ids": selected,
                    "result": result
                })

                does_not_match_count = result.get("summary", {}).get("does_not_match", 0)
                if does_not_match_count > 2:
                    logging.info(f"Shape claim filtered out for class {cls}; does_not_match_count={does_not_match_count}")
                else:
                    keep_shp.append(claim)

            filtered[cls] = {"positions": keep_pos, "shape": keep_shp}

            if save_validation_dir:
                with open(os.path.join(save_validation_dir, f"{cls}_validation.json"), "w", encoding="utf-8") as f:
                    json.dump(class_log, f, ensure_ascii=False, indent=2)

        return filtered

# ------------------------ Orchestrator ------------------------

class TAKTwoAgentPipeline:
    """
    Orchestrates Agent1 (prior generation) and Agent2 (multimodal validation).
    Saves:
      - raw priors  -> save_dir/priors/{cls}.json
      - validation  -> save_dir/validation/{cls}_validation.json
      - filtered    -> save_dir/filtered/{cls}.json
    """
    def __init__(
        self,
        api_key: str,
        image_dir: str,
        metadata_json: str,
        classes_txt: str,
        save_dir: str,
        api_url: str = "https://api.ai-gaochao.cn/v1/chat/completions",
        model: str = "gpt-4o",
        samples_per_class: int = 2,  # per-manuscript: choose 2 cases
    ):
        self.save_dir = save_dir
        ensure_dir(save_dir)
        ensure_dir(os.path.join(save_dir, "priors"))
        ensure_dir(os.path.join(save_dir, "validation"))
        ensure_dir(os.path.join(save_dir, "filtered"))

        self.image_dir = image_dir
        self.metadata = self._load_metadata(metadata_json)
        self.categories = self._load_classes(classes_txt)

        self.client = GPTClient(
            api_key=api_key,
            url=api_url,
            model=model,
            temperature=0.7,
            top_p=0.8,
            presence_penalty=1.0,
            n=1
        )
        self.agent1 = PriorGeneratorAgent(self.client, self.categories)
        self.agent2 = MultimodalValidatorAgentSixViews(
            client=self.client,
            image_dir=image_dir,
            metadata_cases=self.metadata,   # expects case-level structure as described
            samples_per_class=samples_per_class,   # per class: randomly select 2 cases
            max_workers=6,
            views_per_case=6,
            require_exact_six=True
        )

    def _load_metadata(self, p: str) -> Dict[str, List[str]]:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_classes(self, p: str) -> List[str]:
        with open(p, "r", encoding="utf-8") as f:
            cats = [ln.strip() for ln in f if ln.strip()]
        seen = set()
        out = []
        for c in cats:
            if c not in seen:
                out.append(c)
                seen.add(c)
        return out

    def _save_json(self, obj: Any, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    def run(self):
        logging.info(f"Classes: {self.categories}")
        logging.info("Agent 1: Generating priors (text-only) with EXACT templates...")
        priors = self.agent1.generate_all(max_workers=min(8, max(2, os.cpu_count() or 4)))
        # Save raw priors per class
        for cls, data in priors.items():
            self._save_json(data, os.path.join(self.save_dir, "priors", f"{cls}.json"))

        logging.info("Agent 2: Validating priors (multimodal) using 2 cases x 6 canonical views = 12 images per claim ...")
        filtered = self.agent2.validate_and_filter(
            priors,
            save_validation_dir=os.path.join(self.save_dir, "validation"),
            categories_list=self.categories
        )
        # Save filtered priors per class
        for cls, data in filtered.items():
            self._save_json(data, os.path.join(self.save_dir, "filtered", f"{cls}.json"))

        logging.info("Done. Artifacts saved under: %s", self.save_dir)


# ------------------------ Example main ------------------------

if __name__ == "__main__":
    setup_logging()

    # --- REQUIRED inputs ---
    image_dir = "./gpt-generate/metadata"  # full set of labeled images
    # Metadata JSON must be case-level: {"case_id_0001": {"images": ["case1_front.jpg", ... 6 files ...], "classes": ["liver","spleen"]}, ...}
    metadata_json = "./gpt-generate/metadata/metadata.json"
    # Classes to generate TAK for: one per line
    classes_txt = "./gpt-generate/metadata/class.txt"

    # Save directory
    save_dir = "./gpt-generate/metadata/outputs2"

    # API key (reads from env if None)
    api_key = os.getenv("OPENAI_API_KEY", "your_api_key_here")

    # Optional: override API endpoint/model if needed
    api_url = "https://api.ai-gaochao.cn/v1/chat/completions"
    model = "gpt-4o"

    try:
        pipe = TAKTwoAgentPipeline(
            api_key=api_key,
            image_dir=image_dir,
            metadata_json=metadata_json,
            classes_txt=classes_txt,
            save_dir=save_dir,
            api_url=api_url,
            model=model,
            samples_per_class=2
        )
        pipe.run()
    except Exception as e:
        logging.error("Pipeline failed: %s", e)
        traceback.print_exc()
