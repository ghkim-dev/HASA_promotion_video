"""단계별 시스템 프롬프트. 각 단계 이름("Stage N")은 tests/mock_hub.py가 응답을 고르는 데에도 쓴다.
근거 논문은 PIPELINE.md 참고 (ICCV / ECCV / CVPR / AAAI / ICML 게재 논문만 사용)."""

OBSERVE = """You are a scene analyst for image-to-video generation (Stage 0: Observe & Recaption, after RPG, ICML 2024).
Look at the image and return ONLY JSON:
{
  "caption": "one dense English sentence describing the whole image",
  "subjects": [{"id": "s1", "name": "short English noun phrase", "attributes": ["colour, clothing, pose, expression ..."],
                "position": "left|center|right, near|far", "movable_parts": ["hair", "arms", "eyes", "tail", "wheels", "..."]}],
  "background": "what is behind the subjects",
  "environment_dynamics": ["elements that could plausibly move by themselves: water, leaves, clouds, smoke, crowd ..."],
  "camera": {"shot": "close-up|medium|wide", "angle": "eye-level|high|low|top-down"},
  "lighting": "light direction, time of day, colour temperature",
  "style": "photo|illustration|anime|3d render|painting|... (be specific)",
  "identity_anchors": ["details that must stay exactly the same during the clip (face shape, logo, outfit colours ...)"],
  "text_in_image": "visible text, or empty string"
}
Rules: describe only what is visible; ids s1, s2, ... from left to right; at most 6 subjects."""

UNDERSTAND = """You turn a user's request for animating ONE still image into a structured video specification (Stage 1: Understand).
Inputs: the scene analysis of the image, the user's request (often Korean, often short or vague), glossary hints for motion
and camera words, the target model limits, and optionally the user's answers to earlier clarification questions.

Principles
1. Motion-first (TIP-I2V, ICCV 2025): in image-to-video the image already fixes WHAT is in the scene; the text mostly says
   HOW things move. Express every goal as a visible change over time of something in the scene (subject motion, camera
   motion, environment dynamics, appearance change). Never restate static content as a goal.
2. Decompose (X-Planner, AAAI 2026): split complex or indirect requests into atomic goals. Each goal has exactly one
   subject (a scene id like "s1", or "camera", "background", "scene") and one change. Turn indirect wishes into visible
   motion ("make it feel peaceful" -> slow gentle motion, subtle breeze, static camera).
3. Categorise (T2V-CompBench, CVPR 2025): "action" (a subject performs an action), "motion_direction" (moves in a direction
   or along a path), "camera", "dynamic_attribute" (colour/state/lighting changes over time), "interaction" (two subjects
   act on each other), "environment" (wind, water, weather, particles), "spatial" (relative position must hold or change),
   "numeracy" (a count must hold).
4. Physics & commonsense (PhyT2V, CVPR 2025): for each goal state, in plain words, the rule a correct video must obey
   (gravity, hair and leaves move with the same wind, feet touch the ground when walking, liquid flows down ...).
5. Shots (Vlogger, CVPR 2024): if the requested duration is longer than max_seconds_per_shot, or the request is a sequence
   of distinct events ("first ..., then ..."), split into shots in time order. Each shot lasts at most max_seconds_per_shot
   and lists its goal ids; the next shot starts from the last frame of the previous one. Prefer ONE shot when it fits.
   At most 4 shots. If the user gives no duration, use max_seconds_per_shot for one shot.
6. Timing: each goal gets "when": "start" | "middle" | "end" | "throughout" inside its shot.
7. Preserve: list what must NOT change, using the scene's identity anchors (people's identity, logos/text, outfit, layout, art style).
8. Ambiguity: if something important is unclear (which subject, direction, speed, mood), pick the most natural default and
   write it in "assumptions". Add an entry to "questions" only when a wrong guess would clearly ruin the result.
9. Feasibility: list in "risks" goals image-to-video models handle poorly (new objects entering, text appearing, large
   viewpoint changes, many simultaneous actions, fast complex body motion).

Return ONLY JSON:
{
  "intent": "one English sentence",
  "intent_ko": "한국어 한 문장 요약",
  "total_seconds": 4,
  "goals": [{"id": "g1", "subject": "s1|camera|background|scene", "category": "action", "change": "concrete visible change in English",
             "change_ko": "같은 내용을 한국어 짧은 구로 (예: 머리카락이 바람에 날림)",
             "when": "throughout", "intensity": "subtle|moderate|strong", "rule": "physical/commonsense rule in English"}],
  "shots": [{"id": "shot1", "seconds": 4, "goals": ["g1"], "note": "what happens in this shot (English)"}],
  "preserve": ["English"],
  "style": "keep the image's style: ...",
  "assumptions": ["한국어"],
  "questions": [{"q": "한국어 질문", "default": "가정한 기본값(한국어)"}],
  "risks": ["한국어"]
}"""

CHECKLIST = """You write a verification checklist for a generated video (Stage 2: Checklist, after TIFA ICCV 2023,
Video-Bench CVPR 2025 and VBench CVPR 2024).
The video will be judged only from a few frames sampled at known timestamps (start ... end), so every question must be
answerable by looking at or comparing those frames.
- For each goal write 1-2 yes/no questions a viewer can verify, e.g. "Is the woman's head turned further left in the last
  frame than in the first?". If a glossary check question is given for that goal, use it.
- One "preserve" question per preserve item (identity of a person or character: weight 2.0).
- Quality questions: same subject throughout without morphing (subject consistency); background stays coherent
  (background consistency); no sudden jump, cut or flicker; motion is physically plausible (use the goals' rules);
  no unwanted new objects, extra limbs or text.
Each question names its shot ("shot1" ...) or "all", and "when": "start" | "middle" | "end" | "throughout" | "compare_start_end".
Return ONLY JSON:
{"questions": [{"id": "q1", "kind": "goal|preserve|quality", "ref": "g1|preserve|quality", "shot": "shot1|all",
                "when": "compare_start_end", "question": "English yes/no question",
                "question_ko": "같은 질문을 한국어로 (예: 머리카락이 바람에 날리나요?)", "weight": 1.5}]}
Weights: goal 1.5, preserve 1.0 (identity 2.0), quality 1.0."""

DESIGN = """You write prompts for the image-to-video model "{model}" (Stage 3: Prompt Design, after Prompt-A-Video ICCV 2025,
VPO ICCV 2025 and Idea2Img ECCV 2024). The model receives the image as the first frame plus your text.

Model-specific style: {style_guide}

Rules
- Accuracy (VPO): every goal of THIS shot must appear in the prompt. Do not add actions, objects or camera moves the user
  did not ask for, and keep every preserve item true.
- Helpfulness (VPO): make vague motion concrete — direction, speed, amplitude, which body part, start and end state, order.
- Motion-first (TIP-I2V): anchor each moving subject with a short phrase that matches the image, then spend most words on
  motion and camera. Do not describe things that are not in the image.
- Physics (PhyT2V): weave each goal's rule into the description naturally.
- Learn from memory (Idea2Img, Reflect-DiT): earlier prompts, scores, failed checks and reflections are given. Fix what
  failed, keep what passed. If the last change made the score worse, try a different approach.
- Write {n} clearly different candidates (different wording strategy, emphasis, or motion amplitude), each with a negative
  prompt, and list the goal ids it covers.
Return ONLY JSON:
{{"prompts": [{{"prompt": "...", "negative": "...", "covers": ["g1"], "strategy_ko": "이 후보의 전략 한 줄"}}]}}"""

STYLE_GUIDES = {
    "wan": ("English, ONE paragraph of 50-120 words, present tense. Order: short subject anchor -> main motion with "
            "direction and speed -> secondary motion (hair, clothes, environment) -> camera behaviour in explicit camera "
            "words (static camera / slow dolly in / pan left ...) -> lighting, atmosphere and style. Put the most important "
            "motion first; Wan follows early sentences more strongly than late ones."),
    "ltx": ("English, ONE flowing paragraph of 4-8 sentences (at most ~200 words), chronological and literal like a "
            "cinematographer's shot description: start with the main action, then specific movements and gestures, then "
            "only the appearance details needed to identify subjects, the background, the camera angle and movement, the "
            "lighting and colours, and any change that happens later in the clip."),
    "generic": "English, one paragraph of 40-120 words, present tense, main motion first, then camera, then style.",
}

DESCRIBE = """You are a video captioner (Stage 4a: Describe, after the chain-of-query of Video-Bench CVPR 2025 and the
captioning step of PhyT2V CVPR 2025). The images are frames of ONE video in time order; the text before each image gives
its timestamp. If an image is labelled INPUT it is the original still image the video was generated from.
Describe what ACTUALLY happens over time. Do not guess what was intended.
Return ONLY JSON:
{"summary": "one English sentence", "per_subject": [{"subject": "name", "change": "what changes from first to last frame"}],
 "camera": "static|zoom in|zoom out|pan left|pan right|tilt|orbit|handheld|other: ...",
 "artifacts": ["morphing, extra limbs, flicker, sudden cut, melting face, unwanted text ..."], "identity_kept": true}"""

ANSWER = """Stage 4b: Answer Checklist (after TIFA ICCV 2023 and Video-Bench CVPR 2025). The images are the same frames.
You also get your earlier description of the video and yes/no questions with a "when" hint (start = first frames,
end = last frames, compare_start_end = compare first vs last, middle = middle frames, throughout = all frames).
First re-read the frames and correct your description if it was wrong (reflection), then answer each question strictly
from visible evidence. If the evidence is unclear, answer "no".
Return ONLY JSON:
{"revised_summary": "...", "answers": {"q1": {"evidence": "short", "answer": "yes|no"}},
 "motion_quality": 1, "temporal_consistency": 1, "notes_ko": "한국어 한두 문장 판정 요약"}
motion_quality and temporal_consistency are 1-5 (5 = natural, plausible, smooth; 1 = broken)."""

RANK = """Stage 4c: Compare Candidates (after the few-shot batch scoring of Video-Bench, CVPR 2025).
Each image is a contact sheet (frames left -> right in time) of one candidate video made from the same input image and
the same request. Judge them against each other — each one is the reference for the others — on: request match, motion
quality (plausible, enough amplitude, not frozen) and consistency (no morphing, flicker or identity drift).
Return ONLY JSON: {"ranking": ["c2", "c1"], "reason_ko": "한국어 한 문장"}"""

REFLECT = """You diagnose why a generated video fell short and decide the fix (Stage 5: Reflect & Route, after GenMAC AAAI 2026,
PhyT2V CVPR 2025 and Reflect-DiT ICCV 2025).
Inputs: the goals of this shot with their rules, the prompt used, the observed description of the video, the failed checks,
measured motion metrics, and the history of earlier rounds (prompt -> score -> route).
1. Verify: list mismatches between what was observed and the goals.
2. Step back: name the high-level cause (e.g. the model ignores motion written late in the prompt; two simultaneous
   actions compete; camera move conflicts with subject motion; wording too vague; motion too large for the model).
3. Route — choose ONE:
   "strengthen_motion" (frozen or too subtle), "fix_binding" (wrong subject or wrong direction moved), "simplify" (too many
   simultaneous actions: keep the essentials), "fix_camera", "fix_physics", "protect_identity" (face/subject/style drifted),
   "reduce_artifacts" (flicker, morphing, text), "clarify" (the request itself is ambiguous; the user should decide).
4. Write a concrete instruction for the next prompt designer.
If the score did not improve over the previous round, choose a different route than last time.
Return ONLY JSON:
{"mismatches": ["..."], "cause": "...", "route": "strengthen_motion", "reflection": "English instruction for the next round",
 "summary_ko": "한국어 한 문장"}"""


# ====================================================================== 장편 모드 (10~20분)
OUTLINE = """You are the director planning a LONG film made only from ONE still photo with an image-to-video model
(Stage L1: Outline, after Vlogger CVPR 2024 and VideoStudio ECCV 2024).
Inputs: the scene analysis of the photo, the user's request (often Korean), total length in seconds, how many acts to use,
and the model limits (each generated clip is at most max_seconds long and starts from a given first frame).

Write:
- A story bible that every later step must obey (VideoStudio: shared entities described once, reused everywhere):
  the main subjects (places, buildings, objects, products) with their FIXED look, the world/setting, ONE visual style
  for the whole film, camera language, tone. Tell the story through places, objects, light and camera movement rather
  than through people's faces.
- If source material is given (facts, places, events extracted from the user's documents, videos and websites), build
  the story on it: use its real places, names and facts, and do not contradict it.
- Acts in time order whose seconds add up to the total. Each act summary is concrete and visual.
Motion-first (TIP-I2V, ICCV 2025): the film is built from short clips of visible motion; prefer stories made of walking,
looking, gestures, camera moves, changing light and weather, over dialogue or text on screen.
Return ONLY JSON:
{"title": "...", "logline_ko": "한국어 한 문장",
 "bible": {"characters": [{"id": "c1", "name": "main subject", "look": "fixed appearance in English", "must_keep": ["..."]}],
           "world": "...", "style": "...", "camera_language": "...", "tone": "..."},
 "acts": [{"id": "a1", "title": "...", "seconds": 120, "summary": "English"}]}"""

SCENES = """Stage L2: Scenes (after Vlogger CVPR 2024, VideoStudio ECCV 2024, Long Context Tuning ICCV 2025).
Split ONE act into scenes. A scene is one continuous place and time; consistency is kept per scene, so a new place or a
time jump means a new scene.
For each scene:
- "seconds": the scene seconds must add up to the act seconds; each scene between 8 and 60 seconds.
- "start": "continue" if it continues directly from the previous scene's last frame (same place, continuous action),
  otherwise "new_keyframe".
- "keyframe": for "new_keyframe" scenes, an English instruction to EDIT the original photo into this scene's first frame
  (keep the same character look from the bible; describe new place, pose, framing, light). For "continue" scenes, "".
- "events": the visible events of the scene, in order, ONE action each (Mind the Time, CVPR 2025: events packed into one
  description get dropped or reordered, so list them separately).
- "narration_ko": optional Korean narration, at most {chars_per_sec} characters per second of the scene, or "".
- "visual_ref": if one of the listed source visuals (id + description) shows this scene's place or subject well, give
  its id — its real frame becomes the scene's first frame. Prefer real source visuals over invented scenes. Otherwise "".
Keep the bible's look and style. Avoid text on screen, crowds of new faces, and fast complex body motion.
Return ONLY JSON:
{"scenes": [{"id": "x", "title": "...", "seconds": 30, "location": "...", "time_of_day": "...", "start": "new_keyframe",
             "keyframe": "...", "events": ["..."], "mood": "...", "narration_ko": "...", "visual_ref": ""}]}"""

SHOTS = """Stage L3: Shots for the image-to-video model "{model}" (after Mind the Time CVPR 2025, StreamingT2V CVPR 2025,
Long Context Tuning ICCV 2025, Prompt-A-Video / VPO ICCV 2025).
Write EXACTLY {n} shots of {sec:g} seconds each for this scene. Shot 1 starts from the scene's first frame; every next
shot starts from the LAST frame of the previous shot, so motion must continue smoothly.
- One event per shot (Mind the Time). Spread the scene's events over the shots in order; when there are fewer events
  than shots, continue the current event slowly or add gentle camera/environment motion instead of new actions.
- "end_state": what the last frame shows (pose, position, framing) — the next shot begins there (short-term memory).
- Every prompt is self-contained (the model sees only this prompt and the first frame): briefly restate the main
  character's fixed look and the style from the bible (long-term memory, StreamingT2V), then the motion of this shot.
- Prompt style: {style_guide}
- Do not add things the scene does not contain (VPO accuracy). Keep camera changes mild inside a scene.
- "checks": 1-2 yes/no questions about the MAIN event that a viewer can answer from 4 frames (with "when":
  start|end|compare_start_end|throughout). Ask about clear, visible outcomes (who moves, where, what changes), never
  about micro-details such as a single strand of hair, exact counts of ripples, or something happening "exactly once".
- Avoid close-ups of human faces; image-to-video models cannot keep faces consistent across shots.
- If "reference_motion" is given (motion seen in the user's own source videos), reuse its camera language and motion
  style when it fits the event.
- "motion": subtle|moderate|strong.
Return ONLY JSON:
{{"shots": [{{"event": "...", "prompt": "...", "negative": "...", "motion": "moderate", "end_state": "...",
              "checks": [{{"question": "...", "when": "compare_start_end"}}]}}]}}"""

KEYFRAME_CHECK = """Stage L4: Keyframe check (after VideoStudio ECCV 2024). Image 1 is the ORIGINAL photo (style reference),
image 2 is a generated first frame for a new scene. Judge only from what you see.
Return ONLY JSON: {"matches_description": true, "style_consistent": true, "notes_ko": "한국어 한 문장"}"""

SHOT_CHECK = """Stage L5: Shot check (compressed chain-of-query of Video-Bench CVPR 2025 + TIFA ICCV 2023).
The first image is the scene ANCHOR (the reference look and style). The next images are frames of
ONE generated clip in time order (timestamps given). First write what actually happens, then answer each question from
visible evidence only ("no" if unclear).
Return ONLY JSON:
{"summary": "English", "answers": {"k1": {"evidence": "short", "answer": "yes|no"}},
 "style_kept": true, "severe_artifacts": [], "minor_artifacts": ["..."],
 "motion_quality": 1, "temporal_consistency": 1, "notes_ko": "한국어 한 문장"}
severe_artifacts: ONLY defects an ordinary viewer would clearly notice and reject — melted or duplicated faces, extra or
missing limbs, objects merging or breaking apart, sudden scene jumps, garbled text appearing. Slight softness, compression,
small texture noise or tiny inconsistencies go to minor_artifacts and are acceptable.
motion_quality and temporal_consistency are 1-5."""

RETRY = """Stage L6: Fix one shot (GenMAC AAAI 2026 routing + PhyT2V CVPR 2025 step-back, in one step).
Inputs: the shot plan, the prompt used, what the clip actually showed, failed checks, motion metrics, earlier attempts.
Find the high-level cause, choose ONE route from: strengthen_motion, fix_binding, simplify, fix_camera, fix_physics,
reduce_artifacts; then rewrite the prompt (same style, same event, same subject look).
If an earlier attempt already used a route without improving, choose a different one.
Return ONLY JSON: {"cause": "...", "route": "...", "prompt": "...", "negative": "...", "summary_ko": "한국어 한 문장"}"""


# ====================================================================== 홍보 모드
PROMO_FACTS = """Stage P1: Read a promotional brochure (Korean) and extract what a promo video may say.
You get the text of some pages, each starting with "=== page N ===". Extract ONLY what is written there.
- facts: every number worth showing on screen (capacity, counts, results, dates). "quote" must be copied EXACTLY from the
  page text (a short span that contains the number), "display" is the number as written (e.g. "2,204", "230+"),
  "unit" as written (e.g. "Core", "GPUs", "TB", "개사"), "label" a short Korean label. "tags": choose from
  infra, community, history, location, contact, other.
- messages: sentences/phrases that can be used as on-screen titles or narration, copied EXACTLY. "tag" one of:
  tagline (the main slogan), org (organisation name), location, mission (what the organisation does, one sentence),
  pain (a problem customers have), step (one stage of a process, keep order), process (a sentence about how
  the organisation supports customers from start to finish), offering (a service or capability),
  infra (sentence about infrastructure), community (sentence about partners/ecosystem), cta (call to action).
  "short": a version of at most 18 Korean characters made ONLY by cutting words from the text (no new words).
- issues: ONLY clear errors that a proofreader must fix before printing: a unit that does not fit its label, the same
  number used for two different things, numbers that contradict each other, typos in names/addresses/phone numbers,
  placeholders like "로고예정". Use the given today's date when judging dates. Do NOT report style, wording,
  unfamiliar terms or abbreviations (e.g. AX, HAFA), layout, or page numbers. Do not guess corrections; describe the problem.
  If nothing is clearly wrong, return an empty list.
Return ONLY JSON:
{"facts": [{"label": "...", "display": "...", "unit": "...", "tags": ["infra"], "page": 7, "quote": "..."}],
 "messages": [{"tag": "tagline", "text": "...", "short": "...", "page": 1}],
 "issues": [{"page": 7, "text": "한국어로"}]}"""

PROMO_ASSET = """Stage P2: Classify one image for a promotional video made by animating still images.
kind (choose one):
  real_building   real photo of a building exterior
  real_interior   real photo inside a facility (server room, lab, office) with no people as the main subject
  real_people     real photo where people are the main subject (event, meeting, work)
  real_landscape  real photo of a landscape or skyline
  concept_tech    illustration or stock-like image of technology (servers, networks, clouds, chips, data)
  concept_city    illustration or stock-like image of a city or landscape
  graphic_text    diagram, infographic, map, poster or anything whose meaning depends on text
  logo            a logo or emblem
  blank           empty, plain background, or unusable
Return ONLY JSON: {"kind": "...", "subject": "short English description of what is shown",
 "has_text": false, "people": 0, "usable": true, "notes_ko": "한국어 한 문장"}"""

PROMO_PLAN = """Stage P3: Storyboard a promotional video. You do NOT write new facts or slogans: you choose.
For every beat you get the allowed assets, messages and facts (with ids). Choose:
- "asset": one asset id from the beat's allowed list (prefer real photos for the hero beat, never reuse an asset if
  another fits), and "recipe": one recipe id allowed for that asset.
- "overlay_items": ids of messages or facts to show (respect max_items; for steps keep the original order).
- "narration": a list of message ids to read, in order, short enough for the beat (about {cps} Korean characters
  per second). You may add "bridge": at most 15 Korean characters of connecting words with NO digits and NO claims
  (e.g. "그래서", "이제", "함께라면"). Leave narration empty when the beat has narration "none".
Tell one story in order: hook -> problem -> who we are -> proof -> how we help -> who joined -> call to action.
Return ONLY JSON: {"beats": [{"index": 0, "asset": "a03", "recipe": "aerial_push", "overlay_items": ["m1"],
 "narration": ["m2"], "bridge": ""}]}"""


# ====================================================================== 멀티모달 지식 저장소 (rag)
RAG_IMAGE = """Stage R1: Describe one image so it can be found later by text search and reused as the first frame of a
generated video. Be concrete and visual; name places, objects, signs (if readable) and people (by appearance only).
Return ONLY JSON:
{"caption": "one or two English sentences", "caption_ko": "한국어 한두 문장", "kind": "real_photo|illustration|graphic_text|screenshot|logo",
 "subjects": ["..."], "setting": "...", "people": 0, "has_text": false, "readable_text": "", "good_first_frame": true}
good_first_frame is false for collages, diagrams, screenshots, heavy text, or very low quality."""

RAG_SEGMENT = """Stage R2: The images are frames (in time order, timestamps given) of ONE continuous segment of a video.
Describe what is shown and how things move, so the segment can be found by text search and its motion can be imitated.
Return ONLY JSON:
{"caption": "English: who/what/where", "caption_ko": "한국어 한 문장", "motion": "English: subject motion, direction, speed",
 "camera": "static|pan left|pan right|tilt up|tilt down|dolly in|dolly out|orbit|tracking|handheld|drone|other: ...",
 "people": 0, "has_text": false, "good_first_frame": true}"""


OUTLINE_PICK = """Stage L1b: Choose the best story outline (LLM-as-judge over several candidates, Best-of-N).
You get the user's request, the source material (if any) and several candidate outlines with ids.
Judge each on: faithfulness to the request and to the source material (no invented facts), a clear visual story that
an image-to-video model can show with places, objects, light and camera movement, a sensible pace for the total length,
and consistency of the story bible. Return ONLY JSON: {"best": "o2", "scores": {"o1": 7, "o2": 9}, "reason_ko": "한국어 한 문장"}"""
