# CLAUDE.md — "No Word for It": do VLMs ignore what language cannot reach? (initial effect study)

ATML graduate-course research project aimed at NeurIPS (workshop at minimum). This folder holds **project 2**.
Project 1 (length-bias suppression vs safety) was dropped because the full study found the opposite of the effect we hoped to explain; its full history and the reusable Kaggle lessons are in [referance.md](referance.md) (that file is the old project's CLAUDE.md, kept read-only for reference).

- **Team:** the user runs this initial effect study **alone**. Maryam is the user's teammate for the project as a whole.
- Proposal: [VLMs language bias proposal.docx](VLMs%20language%20bias%20proposal.docx) (written by Maryam, 2026-10-24). Background and motivation only; several of its paper descriptions are inaccurate (see §2.3).
- Literature check notes (2026-10-30, four web research passes, every claim linked to its source): [research_notes/VLM unnamed content novelty check/](research_notes/VLM%20unnamed%20content%20novelty%20check/)
- Facts below are **verified** (source read on the date given) unless marked **[unverified]**.

---

## 0. How to work in this folder (read first, always)

This is a research project that will be submitted for review. **There is no room for errors.** Every number, claim and line of code can end up in a paper.

- **Verify before trusting.** Before using any fact (a paper's number, a config value, a library behaviour), check it at the source: the paper's HTML or PDF, the model's `config.json`, the library's code. Write the source next to the fact. If it cannot be checked, mark it **[unverified]** and say so to the user.
- **Double-check every result.** Recompute key numbers a second way (an independent short script, a hand count on a few items, a different library call). Sanity-check against known values (e.g. chance = 25% on 4-way questions; the authors' published accuracies).
- **Test code before it touches real data.** Unit tests on tiny inputs with known answers (token-index mapping, projections, scoring) before any Kaggle run.
- **Never change an evidence rule after seeing the data it judges.** Rules are fixed in §1 before data. Any later change goes in §11 with date and reason, and is reported as a deviation.
- **Report outcomes faithfully.** If a check fails, comes out weak, or points the opposite way, say so plainly. Never drop a condition, item or model silently. A clean negative result is a valid outcome.
- **When unsure, stop and ask the user.** Especially before: anything that costs GPU quota, anything that would need training, any change to the design.
- **Explain in plain words.** The user wants every message in the chat to be easy to understand: short sentences, no unexplained jargon.

---

## 1. The research question and what this effect study must deliver

### 1.1 The idea in one paragraph
Vision-language models (VLMs) often "see" image details but fail to use them, and this is worst for things that have **no name** (a random squiggle, an unnamed point on an object). Inside the language model, each image token is a vector. We can split that vector into a **text part** (the piece that lies along directions the model uses for words) and a **residual** (everything else). **Hypothesis H:** information about *named* things sits in the text part, which the model reads; information about *unnamed* things sits in the residual, which the model ignores.

### 1.2 Why an effect study first
The professor asked us to **first show the effect exists**, and only then explain it. Project 1 failed exactly here: the full study found the reverse effect. So before any patching (RQ2) or fine-tuning analysis (RQ3), this study asks, cheaply, whether the basic pattern H predicts is actually there.

**Deliverable:** one figure plus a graded yes/no answer.
- Panel A: per condition (named shapes, named-complex shapes, squiggles at each complexity), the model's answer accuracy next to its internal "probe" accuracy (chance line at 25%).
- Panel B: per layer, the **text share** S (defined in §6.4) for named vs unnamed, with the random-subspace band.

### 1.3 How the evidence is judged (confirmed by the user 2026-10-30, before any data)
**User decision:** these rules were drafted by Claude, not taken from the literature, so **no number in them is a pass/fail bar**.
- We look for **trends**: does the effect point the way H predicts, does it show up across settings, and roughly how big is it.
- **A moderate, consistent effect is enough to continue.** An effect does not have to clear a high predefined threshold.
- Specific numbers in this plan (k = 64, 600 items per set, 100 calibration items, 100 random subspaces, the complexity measure) are sensible defaults. They may be tuned **before** data, with a §11 entry.

Every quantity is still reported with its size and a 95% bootstrap CI (10,000 resamples over items; clustered by shape identity where shapes repeat). The CI shows how sure we are; it is not a gate. Each check gets one grade:
- **Clear:** right direction, CI excludes 0, and it holds across most settings (layers, k, text definitions, models).
- **Moderate:** right direction in most settings and visible in the plots; the CI may touch 0 in some settings. **Enough to continue.**
- **Weak:** right direction but small or inconsistent across settings. Reported, and worth continuing only together with other checks that are moderate or better.
- **Absent:** no consistent difference.
- **Reversed:** a consistent difference in the opposite direction (what happened in project 1). Always said plainly.

**Two things stay strict, because looser rules make it easier to fool ourselves with noise:**
1. The main view of each check (written below) is fixed before the data.
2. Every setting is shown. We never pick the best-looking layer, k or definition afterwards and report only that.

**The checks (what H predicts):**
- **E1. The behaviour gap exists in our setup.**
  - (a) Named basic shapes are answered better than squiggles (n = 30). Main view: Acc(named) − Acc(squiggles n=30).
  - (b) The gap is about names, not about fill, complexity or look-alike shapes. Main view: the core 2×2 table (§5.1). Named beats unnamed **within** filled shapes and **within** line drawings. Backed by the `named` coefficient in `correct ~ named + filled + log(complexity) + lookalike`, clustered by shape identity (covariates in §5.3).
  - (a) absent or reversed → the effect is not there in our setup; stop. (b) absent → the gap comes from how the shapes look, not from names; report it, and the "no word" framing weakens.
- **E2. Unnamed information is encoded but not used.**
  - (a) Squiggle information is inside the model: probe accuracy above chance (25%).
  - (b) **Main view:** among items the probe answers correctly, the model still answers named items better than unnamed ones. Matching item by item removes the "unnamed is just less encoded" explanation.
  - (c) The utilisation gap U = probe acc − behaviour acc is larger for unnamed than named.
  - (a) or (b) absent → "not encoded" rather than "encoded but ignored"; H's premise weakens.
- **E3. The core location test.**
  - Named information sits in the text part more than unnamed information does: S(named) > S(unnamed).
  - The text part beats random subspaces of the same size for named content: S(named) > 0.
  - Main view: k = 64, S averaged over post-injection layers (§6.4). All layers, k values and all three text definitions (§6.3) are shown next to it, to see whether the trend is consistent.
  - Absent or reversed → the simple geometric split is wrong; report a careful negative.
- **E4. Removal check (supporting).** Removing the top text directions from image tokens lowers named accuracy more than unnamed accuracy, relative to removing random directions of matched projected norm.
  - If unnamed accuracy instead **rises**, that supports the competing "text part masks the residual" story of Saini et al. (§2.2). Report it: it is a different paper, not a failure to hide.
  - Main view (fixed 2026-10-31, before data): Qwen only, k = 16, the four core 2×2 sets. Drop = Acc(no removal) − Acc(removal). Effect = [drop(T2) − mean drop(random)] for named minus the same for unnamed, averaged over filled and line. k = 2 and each 2×2 cell are shown next to it.
- **E5. Validity of the measure (recommended, no training needed).** On the authors' released *ordinary-names* checkpoint (§3), S rises for the **taught** squiggles (`taught30`) more than for the **untaught** ones (`untaught30`), and more than in the base model. If teaching names does not move squiggle information into the text part, the S measure probably does not measure what we think; say so next to E3.

**Overall decision (judged from the whole trend picture, not one number):**
- **Go** to the full project (RQ2 patching, RQ3 fine-tuning analysis) if E1(a), E2 and E3 are at least **moderate** on Qwen3-VL-2B, and Gemma3-4B points the same way (at least weak, not reversed).
- **Discuss with the user** (and with Maryam and the professor) if the core checks are only weak, or if the two models disagree.
- **Stop or pivot** if E1(a) or E3 is absent or reversed.

### 1.4 Out of scope until the go decision
RQ2 (swapping only the text part or only the residual between two images), RQ3 (what fine-tuning changes), HSIC as a headline measure (it is a secondary measure in Step 7), LLaVA-1.5-7B (optional, last).

---

## 2. What is already known (literature check 2026-10-30)

### 2.1 The six papers the proposal cites (all exist; details verified on arXiv)
| # | Paper | arXiv | What it actually shows |
|---|---|---|---|
| 1 | Fu, Bonnen, Guillory, Darrell, *Hidden in plain sight: VLMs overlook their visual representations* | 2506.08008 (Jun 2025; no venue on arXiv) | VLMs fall far below a readout of their own vision encoder on correspondence, matching, affordance (e.g. semantic correspondence 0.536 vs 0.225). Information stays probe-able in the LLM; the LLM is the bottleneck. LoRA on the LLM helps most. Code: github.com/stephanie-fu/hidden-plain-sight |
| 2 | Veikherman et al., *Encoded but Unused: Bounding the Sources of VLM Errors* | 2610.29241 (21 Oct 2026) | 84.4% of LLaVA-1.6-7B and 70.0% of Qwen2.5-VL-7B errors are "encoded but unused" (lower bounds 77.5 / 61.0). Only nameable attributes (colour, material, shape). Its probe reads through the **vocabulary** (logit lens), so the unused information it finds is text-aligned by construction. No code yet |
| 3 | Shahgir et al., *VLMs Need Words: VLMs Ignore Visual Detail In Favor of Semantic Anchors* | 2604.02486 (v3 4 Aug 2026; COLM 2026) | Same tasks and models as ours. Qwen3-VL-2B squiggles: answer 29.0, probe 74.2; name teaching → 86.0; task fine-tuning → 99.3. Already has a MaxSim probe and logit-lens analysis. **Code, data and checkpoints are public** (§3, §4) |
| 4 | Ou et al., *When Vision Becomes Text* (SIEVE) | 2608.10489 (11 Aug 2026) | Splits visual tokens into a projection on the span of the ~30 post-image **prompt-token** states + residual; uses only the residual norm, for token pruning. The authors say *text tokens absorb visual content* as depth grows. No code |
| 5 | Saini, Jha, Tang, Liu, *When Language Overwrites Vision: Over-Alignment and Geometric Debiasing in VLMs* | 2605.08245 (v4 23 Jun 2026) | Text subspace = top principal components of **caption embeddings**. Removing them from vision tokens lowers hallucination (LLaVA AMBER CHAIR 9.20 → 6.74) and raises probe decodability. Treats the text part as *harmful bias*. Qwen2.5-VL-7B, LLaVA-1.5-7B. No code |
| 6 | Salazar et al., *Pathways of Visual Information Flow in VLMs* | 2607.03358 (NeurIPS 2026 poster) | Interchange patching of whole token groups (image, query, last token). No subspace patching. Code: github.com/israfelsr/vlm-pathways |

### 2.2 Other papers we must cite
- **RL Teaches VLMs Where to Attend but Not How to Perceive** (2610.23983, 14 Oct 2026). Fine-tuning gains sit in the LLM; image attention rises only slightly (0.340 → 0.358); attention-only fixes recover just 1–5 points. Closest competitor for RQ3.
- **The Count Is There, but Misaligned** (2607.09544). Counting information sits in directions the readout does not use. Nearest "unused subspace" result.
- **LatentLens** (2602.00462, ICML 2026). The logit lens badly undercounts how word-like visual tokens are (about 24% vs 68% interpretable). A "no words found" result from the logit lens alone is weak evidence.
- **Visual Instruction Tuning Aligns Modalities through Abstraction** (2606.03871). Tuning aligns visual features with existing text features in middle layers; the opposite direction to H for RQ3.
- **Visually Prompted Benchmarks Are Surprisingly Fragile** (2512.17875). Marker colour or size alone can flip results, so marker style must be identical across conditions.
- **Makelov et al. 2023** (2311.17030). Subspace activation patching can create illusory effects; relevant for RQ2 later.

### 2.3 Corrections to the proposal (fix before it is shown to anyone)
1. "VLMs Need Words … through a mechanism the authors could not explain" is **wrong**. The authors state one: fine-tuning "teaches a distinct mechanism — direct visual comparison — that does not route through language". Correct wording: *they give a behavioural account but do not locate it in the model's representations.*
2. SIEVE's "text space" is the span of the prompt's own text-token states, not vocabulary directions, and the authors read the trend as text tokens absorbing vision, not image tokens drifting to text.
3. Saini et al. says roughly the **opposite** of H: the text part is the harmful part, and the residual becomes more usable once the text part is removed. The project must tell "residual ignored" apart from "residual masked by the text part" (E4 does this cheaply).
4. Encoded but Unused covers only nameable content and detects information through the vocabulary, so it does not support "unused information sits in the residual".
5. The "29% → 86% / 99%" numbers are all Qwen3-VL-2B, direct answers, squiggles (not keypoints). 86% comes from teaching names on separate single-image tasks; 99.3% from training on the task itself (out of distribution it drops to 61–76%).
6. The authors' fine-tuning was **full fine-tuning of all parameters** (vision encoder included), not LoRA (§4).

### 2.4 Novelty status (as of 2026-10-30)
- **Already done, do not claim:** the named vs unnamed behaviour gap; "encoded but not output"; logit-lens labels for named content; splitting vision tokens into a text part + residual (Saini, SIEVE).
- **Open:** how much task information sits in the text part vs the residual, **named vs unnamed**, against random subspaces of the same size, under several text-subspace definitions (RQ1 / this study); swapping only one part between images (RQ2); what fine-tuning changes at the subspace level (RQ3); linking human nameability scores to the residual share.
- **Framing:** "the mechanism behind VLMs Need Words".
- **Speed risk:** three closely related papers appeared May–Oct 2026. Re-check the Semantic Scholar citations of 2604.02486 before writing up.

---

## 3. Models (pinned; verified 2026-10-30 from the Hugging Face API)

| id | Repo | Revision | Role | Notes |
|---|---|---|---|---|
| `qwen` | `Qwen/Qwen3-VL-2B-Instruct` | `89644892e4d85e24eaac8bacfd4f463576704203` | main model | 28 layers, hidden 2048, vocab 151,936, **tied** embeddings, DeepStack at vision layers [5, 11, 17], patch 16, merge 2 (one LLM token = 32×32 px). Open |
| `gemma` | `google/gemma-3-4b-it` | `093f9f388b31de276ce2de164bdc2081324b9767` | second model | Gated; **access granted** (HF page says so; local token `Mohidali1245` passes `auth_check`, 2026-10-30). From the official config at this revision: 34 layers, hidden 2560, 256 image tokens per image, 896-px images, patch 14. Vocab size and tied embeddings are not in the config file; loading it with transformers 4.57.1 gives vocab 262,208 (tokenizer length 262,145) and **tied** embeddings (library defaults; re-check under Kaggle's transformers 5). **Overflows in fp16** (NaN / empty output): run in fp32 across both T4s |
| `llava` | `llava-hf/llava-1.5-7b-hf` | `b234b804b114d9e37bb655e11cbbb5f5e971b7a9` | optional, last | Vicuna-7B (32 layers, hidden 4096), vocab 32,064, 576 image tokens (336 px, patch 14). Likely near chance on these tasks |
| `qwen_names` | `Lancelot53/qwen3-vl-2b-instruct-ordinary-names` | `d4b5086ade03fd8fc5805586d03f49e7edd346f1` | E5 validity check | authors' name-taught model; full weights, single `model.safetensors` (4.26 GB); **no processor files: load the processor from `qwen` at its pinned revision** |
| `qwen_task` | `Lancelot53/qwen3-vl-2b-instruct-task-finetuned` | `b2022d30302e94be141018006860f0db46503051` | RQ3 later (not in this study) | authors' squiggle-fine-tuned model; full weights |
| `gemma_names` | `Lancelot53/gemma3-4b-it-ordinary-names` | `b24887f551cfb4eaeec6d4a9a96e17eeff3e5a9b` | E5 on Gemma (optional) | |
| `gemma_task` | `Lancelot53/gemma3-4b-it-task-finetuned` | `9c7e2891b13d9c7cecf8603ccd8dfa0617937cd8` | RQ3 later | |

The released checkpoints mean **RQ3 can likely be studied without any training** (compare `qwen` vs `qwen_task`).

---

## 4. The authors' code and data (verified 2026-10-30 by reading the repo)

Repo `github.com/Patchwork53/VLMs-Need-Words-COLM2026`, branch **`master`**, commit `18655d407bbe69864eea3d2f13bba0144c8f4e33` (2026-08-04). `SeveredAsif/VLMs-Need-Words-COLM2026` is a fork of it. No licence file.

**Task (2D shapes)**, from `2D_shape_recognition/generate_dataset.py` and `data_class.py`:
- Two images per item. Each image has the **same 4 shapes** at random positions (min distance 0.8 in a 4×3 data area, shape half-size 0.3). Image 1 labels one shape "REF" in bold red text below it; image 2 labels all four "A"–"D".
- Matplotlib figure 8×6 in, default dpi 100, tight crop. One random dark colour for shapes and one random light colour for the background **per image** (red labels kept).
- The shapes in image 2 are **pixel-identical copies** of those in image 1 (no rotation, scale or warp); only position and colour change. So the task is "find the identical shape".
- Prompt (exact): `Which shape in the second image is most similar to the REF shape in the first image?\nSelect from the following choices.\n(A) Point A\n(B) Point B\n(C) Point C\n(D) Point D\n`
- Direct answers: the assistant reply is pre-filled with `The correct option is (` and up to 5 tokens are generated greedily. CoT uses the suffix `Think step by step before selecting an answer.` and Qwen3-VL-8B to extract the letter.
- Their code runs in **bf16 with flash-attention 2**. T4 has neither, so our numbers may differ slightly; this is a stated limitation.

**Stimulus files:**
- `basic_shapes/` holds **8** known shapes (circle, diamond, hexagon, oval, pentagon, square, star, triangle), 512×512 RGBA, **filled solid silhouettes**. The paper says "10 common shapes"; the released test set uses exactly these 8 (49–57 uses each), so the paper's 10 is most likely a text error.
- Each `squiggles_{20,30,40,50,100}/` folder holds only **10** squiggle images. `squiggles_30_v2/` holds 10 *different* n=30 squiggles.
- **Squiggles are thin open lines, not filled shapes.** Generator `generate_unknown_shapes.ipynb`: a random walk of n anchors (step 30–100 px, within radius 100 of the centre, ≥ 15 px from every earlier anchor) on a 200×200 canvas, a cubic spline through them, drawn as a line of width 4 px (colour 20,20,20), 4× supersampling. **No random seed is set**, so the exact squiggles cannot be regenerated; only the saved PNGs can be reused.
- Seen by eye (2026-10-30): the 4 named shapes in an item look very different from each other; the 4 squiggles look alike (round tangles). So the authors' named vs unnamed contrast also differs in **filled vs line** and in **how similar the 4 shapes are**. §5 controls both.
- **Test-set sizes resolved:** the released `basic_shapes_TEST/` and `squiggles_30_TEST/` have **100 items** each (answer letters roughly balanced, 23–31 per letter). The paper's numbers come from **1,000-item** sets that are not released: exact decimals in every saved probe result file are multiples of 1/1000 (the "SANITY" files: 8 items).
- Released Qwen direct-answer result on the 100-item squiggle set: **32%** (paper, 1,000 items: 29.0). It rarely answers "A" (8 of 100 answers; 24 are truly A), so there is a **letter bias**.

**Probe** (`rep_probe_qwen.py`):
- Parameter-free MaxSim. Take the hidden states of the LLM tokens covering the correct REF shape's box and of each of the 4 target boxes. Average over REF tokens of the best cosine similarity to any target token, then pick the highest-scoring target.
- Done at every layer of `output_hidden_states`; the paper reports the best layer. The box→token mapping is `find_qwen3vl_image_tokens` (32-px blocks).
- Read 2026-10-31: the probe scripts pass the PIL images straight to the processor and use **no pre-fill**. The Qwen evaluation script instead runs images through `qwen_vl_utils.process_vision_info` first, which (depending on its version, **[unverified]** which one they used) resizes to a multiple of 28 px, e.g. 812×588, then 800×576 inside the processor: 450 tokens per image instead of 475. So their behaviour and probe numbers may come from slightly different image sizes. We use the probe path for both (one forward pass). The pre-fill does not change the image-token states, because they come before it and attention is causal.
- **Saved probe results (read 2026-10-30 by disassembling the pickles; they were never loaded):**
  - Qwen3-VL-2B, released 100-item squiggles: best 75% at index 4. Already 65% at index 0. It **falls to 38–50% in the middle layers** and ends at 51%.
  - Qwen3-VL-2B, 1,000-item squiggles, file `QWEN_taught_names/…namesbase`: best **74.2 at index 4**. This is the paper's number.
  - **But** `squiggle_rep_results/QWEN/msize2B_datasets30_TEST` (also base Qwen, also 1,000 items) gives best **90.4 at index 25**. Same model, same kind of test, very different number: the authors' probe result depends strongly on details we cannot see. **Never compare our absolute probe numbers to theirs; compare conditions within our own fixed method.**
  - Gemma3-4B (pan-and-scan on): basic shapes 98.9 (index 4), squiggles 91.7 (index 3). These match the paper.
  - **Name teaching (Qwen, ordinary names): squiggle probe 74.2 → 99.5, and the best layer moves from index 4 to index 25.** Human names 98.9, random strings 93.3, "similar" names 89.6. Gemma ordinary names: 91.7 → 93.9. This is an early hint in the direction H predicts (relevant to E5).

**Name teaching** (`teaching_arbitrary_names/teach_names_qwen.py`):
- Full fine-tuning (`FREEZE_VISION_AND_PROJECTOR = False`), 1 epoch, lr 1e-5, seed 42, 10,000 augmented samples per squiggle, single-image tasks only.
- **The 10 taught squiggles are byte-identical to the 10 test squiggles** in `2D_shape_recognition/squiggles_30/` (sha256 checked). `squiggles_30_v2/` holds 10 *untaught* squiggles from the same generator: a ready-made control for E5.

**Task fine-tuning:**
- Qwen (`finetune_qwen_on_task.py`): **full fine-tuning; the freeze function is commented out, so the vision encoder and merger are trained too.** AdamW, lr 1e-5, 4 epochs, batch 8, grad clip 1.0, bf16 autocast, loss only on the answer letter. Data `squiggles_30_TRAIN` (not in the repo); best epoch chosen on `squiggles_30_VAL`.
- Gemma (`finetune_gemma_on_task.py`): **only the language model is trained** (`'language_model' in name`); vision encoder and projector frozen; 4 epochs, lr 1e-5, pan-and-scan on. **So the two released task-fine-tuned models were trained differently** (matters for RQ3).

**Gemma settings:** `do_pan_and_scan` is a required flag in their eval and probe scripts; their saved Gemma results are named `psTrue`, so the paper used pan-and-scan **on**.

**Token counts** (measured 2026-10-30 with the pinned processors, transformers 4.57.1, on released item 0, authors' prompt and pre-fill):
- Images are 800×600.
- Qwen: grid 38×50 patches per image, so **475 LLM image tokens per image**, 950 per item, 1,011 tokens in total. A 120×120-px shape box covers about 16–25 tokens.
- Gemma, pan-and-scan **off**: 256 per image, 512 per item, 582 in total.
- Gemma, pan-and-scan **on**: each image becomes 3 crops of 896×896 (whole image + 2), so 1,536 image tokens per item, 1,660 in total.
- Gemma's processor warns that it defaults to the *slow* image processor under 4.57 and the fast one in newer versions: **always set `use_fast` explicitly** and record it.

**SPair keypoints:** named / no-name tags are inside `semantic_correspondence/create_sem_corr_dataset.ipynb` (`kps_tags`, filters `named` / `no-name`). Not used in this study (later, optional).

**Local copy:** sparse clone (only `2D_shape_recognition/`, `teaching_arbitrary_names/`, `semantic_correspondence/` code, README) at the pinned commit in `data/raw/vnw_repo` (19 MB, 637 files). Git on this machine has `core.autocrlf=true`, so text files there have CRLF line endings; their content is otherwise byte-identical to GitHub (checked). Never run code from inside that folder.

---

## 5. Stimuli for the effect study (built locally in Step 2)

### 5.1 Conditions
**Why the design changed (Step 1, 2026-10-30):** in the authors' sets, "named" shapes are filled, simple and easy to tell apart, while "unnamed" squiggles are thin lines, complex and look alike. Any of these three differences could cause the gap. So the core is a **2×2 design** (has a name × filled or line), plus a complexity sweep and covariates.

| Set id | What | Items | Purpose |
|---|---|---|---|
| `rep_named` | authors' released `basic_shapes_TEST` | 100 | exact replication |
| `rep_squig30` | authors' released `squiggles_30_TEST` | 100 | exact replication |
| `named_fill` | the authors' 8 shape types (circle, oval, square, diamond, hexagon, pentagon, triangle, star) **redrawn by our renderer** with the authors' proportions, filled | 600 | core 2×2: named, filled |
| `named_line` | the same 8 shapes drawn as **outlines** (4-px line, same as the squiggles) | 600 | core 2×2: named, line |
| `unnamed_line` | 40 new squiggles, n = 30, authors' random-walk recipe with a seed | 600 | core 2×2: unnamed, line |
| `unnamed_fill` | 40 random smooth **closed, filled** blobs (periodic spline through 8 anchors, radius 0.35–1) | 600 | core 2×2: unnamed, filled |
| `named_cx` | 10 familiar, more complex filled icons: heart, arrow, crescent, plus, lightning, cloud, key, teardrop, spade, club | 600 | named at higher complexity |
| `squig_n{5,10,20,50}` | 40 line squiggles at each other complexity level | 600 each | unnamed, complexity sweep |
| `taught30` | the authors' 10 name-taught squiggles (`squiggles_30/`), our generator run | 600 | E5: taught |
| `untaught30` | the authors' 10 untaught squiggles (`squiggles_30_v2/`), our generator run | 600 | E5: untaught control |
| `kilogram` (optional) | KiloGram tangrams, bottom-100 vs top-100 naming divergence | 600 | human-normed nameability axis; licence **[unverified]** |

- **Never use letter shapes or digits.** They would clash with the A–D answer labels.
- `named_fill`, `named_line` and `named_cx` shapes must pass a **model naming check** (Step 4 smoke run): the model names each one correctly from a single image. A shape the model cannot name moves out of the named sets (reported).
- **How shapes are drawn** (`src/nwf/shapes.py`, every non-authors shape):
  - 200×200 white canvas, colour (20,20,20), 4× supersampling, 4-px lines, as in the authors' squiggle code.
  - **Every shape is scaled so its longer side is 180 px**, matching the authors' squiggles (180–194 px). So the 2×2 cells differ only in name and fill.
- **How items are laid out** (`src/nwf/stimuli.py`):
  - The authors' `save_image` and `change_colors` ported line by line (same figure, labels, two JPEG saves, box maths).
  - A test runs the authors' own functions (taken from their script) next to ours: **the JPEG bytes and boxes are identical** on 4 cases (2026-10-30).
  - The only change is in placement: centres are kept **0.3** from the left and right edges instead of the authors' 0.25. With 0.25 a shape could stick out of the picture by up to 10 px and be partly cut off. This happens in the authors' released items too: 23 of 800 boxes in `basic_shapes_TEST` (22 items) and 21 of 800 in `squiggles_30_TEST` (19 items), at most 10 px outside. Our sets never cut a shape; the `rep_` sets are kept exactly as released.
- Every random choice is seeded (one seed per set in `config.SETS`; item *i* uses the generator `[seed, i]`), so `python -m nwf.stimuli` rebuilds byte-identical files (tested by building twice).
- The correct letter is **exactly** balanced (each letter is the answer for a quarter of the items).
- `data/stimuli/manifest.json` holds per set: counts, letter counts, the sha256 of `items.jsonl`, and one sha256 over the listing of every file's sha256. Re-checked on Kaggle.
- Item row fields: `id, set, shapes, ref_index, answer, ref_image, tgt_image, ref_boxes, tgt_boxes, named, filled, complexity (one per shape), lookalike, calibration`. Boxes are `[[x0,y0],[x1,y1]]` in pixels of the 800×600 image, in the order of `shapes`. **The `rep_` boxes can reach outside the image; clip them before use.**

### 5.2 Splits
- In every set, the first 100 items (by id) are a **calibration split**. They are used only to choose the probe layer (§6.2) and are excluded from all evidence tests.
- Sets with a pool of 40 shapes (blobs and squiggles): the pool is split into two halves of 20 (even and odd shape numbers), and item *i* draws its 4 shapes from half *i* mod 2. A linear probe trained on one half's items is tested on the other half's items, so it never sees a test shape.
- Sets with 8 or 10 shapes (named, icons, the authors' squiggles) draw from the whole pool; there a linear probe can only be split by item, which is stated as a limitation.

### 5.3 Covariates (fixed before data, revised once in Step 2 before any model data; computed from the shape images, never from model outputs)
- **Complexity:** perimeter² / (4π · area) of the shape's ink at 200×200. The perimeter counts ink-to-background changes between neighbouring pixels (up, down, left, right). It is lowest for a filled circle (about 1.3 with this pixel count) and grows with complexity. Line drawings are much higher (outlines about 70; squiggles n=30 about 170–200; n=5 about 55–75). Used as `log(complexity)`.
- **Look-alike:** how similar the **closest wrong answer** is to the REF shape (target and distractor similarity, as in visual-search research).
  - Each shape gets a signature: its ink mask cropped, padded to a square with a 50% margin (keeps proportions; a filled square still has an edge), resized to 64×64, blurred (Gaussian σ = 3) and mean-centred.
  - Look-alike = the highest correlation between the REF signature and the 3 wrong-answer signatures.
  - This is **coarse** shape similarity, roughly at the model's token resolution (a shape spans only about 4×4 Qwen tokens). Filled convex shapes (circle, oval, pentagon) score as quite alike under it.
- **Filled:** 1 for filled shapes, 0 for line drawings.

---

## 6. Measurement conventions (fixed)

### 6.1 Running the models
- Hugging Face transformers with plain PyTorch hooks. No vLLM (we need hidden states, not throughput).
- `attn_implementation="sdpa"` (T4 cannot run flash-attention 2).
- dtype (rule fixed 2026-10-31, before any model data; record the dtype used on every run):
  - Candidates: `qwen` whole model in fp16 (fits one T4); `gemma` vision tower (SigLIP) in fp16 with the projector and language model in fp32, across both T4s. Fallback for each: fully fp32.
  - **Precision check in the smoke run:** the same 260 smoke items (20 per set) go through each model twice, once fully fp32 and once in the candidate mode, and are compared set by set. The candidate mode is used only if **all three** hold:
    1. no NaN or inf anywhere (vision output, hidden states, letter logits);
    2. the image tokens entering the language model are nearly identical in both modes: cosine similarity ≥ 0.999 on every item (for Qwen this includes the DeepStack features);
    3. the predicted letter is the same on ≥ 98% of the 260 items, and every flip is a near-tie: in the fp32 run the top two letter probabilities (softmax over the four letter logits) differ by < 0.05.
  - If any condition fails, that model runs fully in fp32 and the hours are re-planned.
  - Reported, not gating: per-layer cosine of the region hidden states and MaxSim probe agreement between the two modes. The agreement numbers go in the paper.
  - Why fp16 is acceptable at all: the authors' published numbers come from bf16, which rounds more coarsely than fp16; every condition gets the same setting, so rounding cannot create a named vs unnamed difference; the only real risk is overflow, which the check catches.
- Greedy, no system prompt, the model's own chat template, `add_generation_prompt=True`, then append the authors' pre-fill `The correct option is (`.
- Gemma: pan-and-scan **on** (as the authors). Image processor `use_fast` set explicitly and recorded (the default changed between transformers versions).
  - Each repo declares its own image processor type, and we use that one: **Qwen `use_fast=True`** (`Qwen2VLImageProcessorFast`), **Gemma `use_fast=False`** (`Gemma3ImageProcessor`, PIL). Checked 2026-10-31: each type gives byte-identical pixels under transformers 4.57.1 and 5.16.1 (in 5.x the classes are renamed and `use_fast` is deprecated in favour of `backend=`, but still works). Fast and slow differ by at most one 8-bit level.
  - Gemma's pan-and-scan settings (min crop 256, max 4 crops, ratio 1.2) are the processor's own defaults in both versions (the config file leaves them empty); the authors' crop function matches the library's.
  - Gemma's chat template trims the trailing newline of the prompt; Qwen's keeps it. Same for the authors.
- Token counts per item, measured 2026-10-31 on both versions: Qwen 1,011 (grid 38×50 per image), Gemma 1,660 (6 crops of 896 px). The last input token is ` (` (320 / 568) in both.
- **Scoring:** one forward pass; read the next-token logits for the four letter tokens `A B C D` and take the argmax.
  - Verified 2026-10-30 (tokenizers at the pinned revisions): after `The correct option is (` each letter is one token. Qwen: ` (` = 320, A–D = 32, 33, 34, 35. Gemma: ` (` = 568, A = 236776, B = 236799, C = 236780, D = 236796. Re-check under Kaggle's transformers version.
  - Check in Step 4 that the argmax agrees with the authors' 5-token greedy generation on the smoke items.
  - Because the model has a letter bias (§4), always report the share of each predicted letter next to accuracy.
  - Also keep the full four-letter probability vector.
- Behaviour and hidden states come from the **same forward pass**.

### 6.2 Hidden states
- Captured with our own `forward_pre_hook` on every decoder layer, so stream index *s* = the exact input to decoder layer *s* (s = 0…L−1), plus *s* = L = the input to the final norm.
- **Do not rely on `output_hidden_states`; what it records depends on the transformers version** (measured 2026-10-31 on tiny models built from the real configs, `tests/test_hooks.py`):

  | Version | Model | Recorded index 1–3 vs our hooks | Recorded last index |
  |---|---|---|---|
  | 4.57.1 | Qwen | equal (DeepStack edits the stored tensor in place, so it already holds the extra features) | before the final norm |
  | 4.57.1 | Gemma | equal | after the final norm |
  | 5.16.1 | Qwen | **differ** by exactly the DeepStack features at image positions (5.x copies before adding) | after the final norm |
  | 5.16.1 | Gemma | equal | after the final norm |

  So the authors' own Qwen probe numbers depend on which transformers version they ran. Our hooks give the same states in both versions.
- For Qwen, **post-injection layers are s ≥ 3**. s = 0–2 are reported separately.
- Region tokens: the authors' box→token mapping, ported and tested (`tests/test_tokens.py`): equal to the authors' own functions on real items (including boxes past the picture edge), and checked against the processors' pixels with a drawn white box.
  - Qwen: a token is included if its 32-px block overlaps the box (partly covered edge blocks count).
  - Gemma (pan-and-scan on): **only the crop tokens** are used, not the global view (as the authors). The authors' rule leaves out the blocks the box only partly enters on its right and bottom side. Kept as is, since we compare only within our own method.
- **Storage (decided 2026-10-31): every probe score is computed on the GPU right after each forward pass; hidden states are never stored.** Saved per item: the four letter logits, the top token, MaxSim scores of the 21 named views (§6.4) × 4 targets at every stream index for the raw vectors and for the vectors with massive dimensions removed, the answer picked in each of the 100 random subspaces (4 sizes × inside/outside), the mean vector of each of the 5 regions per stream (fp16, for the linear probe), and token norms. About 0.7 MB per item for Qwen, 1 MB for Gemma. Resumable in chunks of 100 items.
- **Probe layer for E2(b):** the single stream index with the highest pooled (all conditions) MaxSim accuracy on the **calibration split**, fixed before the evidence tests.

### 6.3 Text-subspace definitions (all three are always computed)
- **T1, vocabulary:** top-k right singular vectors of the **mean-centred** unembedding matrix. For Qwen this equals the input embedding (tied). The full vocabulary spans the whole space, so only a top-k version is meaningful.
- **T2, caption text:** top-k principal components of text-token hidden states at the same stream index. These come from a fixed caption set (2,000 COCO captions, seed 0, sources and hashes recorded) passed through the same model, as in Saini et al.
  - Source (user decision 2026-10-31): `yerevann/coco-karpathy` at commit `448fdb1b`, file `data/validation-00000-of-00001.parquet` (standard COCO validation split, 5,000 images × 5 captions; sha256 `1cf26312…`). `python -m nwf.captions` sorts by COCO id, picks 2,000 images with seed 0, keeps each one's first caption, and writes `assets/captions.txt` (committed; 7–24 words each, 1,995 distinct strings) plus `assets/captions_source.json` with both hashes.
- **T3, prompt span:** span of the item's own post-image prompt-token states at the same stream index (SIEVE-style), via regularised least squares as in SIEVE.

k ∈ {16, 64, 256}; primary k = 64.

**How each is computed (fixed 2026-10-31, before any model data; `run_items.fit_bases`, `maxsim.py`):**
- All bases are orthonormal and nested: the top-k basis is the first k columns of the top-256 basis.
- T1: top-256 eigenvectors of the centred unembedding Gram matrix (fp64). Same for every stream index.
- T2: per stream index, top-256 principal components of the caption-token states (all caption tokens except position 0, where language models park huge activations), from the model's own forward pass over the captions as plain text.
- Image-token PCA baseline: per stream index, top-256 principal components of **all** image tokens of the first 20 items of every set (260 items, all inside the calibration split).
- Covariances are summed in fp64 after shifting by the first batch mean (avoids cancellation).
- T3: SIEVE exactly as in the paper (2608.10489, read 2026-10-31): centre on the prompt-token mean t̄, Gram G = TcTcᵀ, ridge λ = the eigenvalue where the running energy first reaches η = 0.75 (their best setting), plus a tiny 1e-6·σ₁²; text part = vc Tcᵀ(G+λI)⁻¹Tc, residual = vc − text part. Prompt tokens = every token after the last image (prompt, template and pre-fill).
- Random: 100 subspaces from seeded Gaussian draws (`default_rng([0, i])`) orthonormalised by QR, the same for every stream index. T3's random baseline uses size Nt − 1 (its rank) applied to the same t̄-centred vectors.
- The "outside" (residual) MaxSim is computed exactly from full dot products minus inside dot products (named views in fp64), tested equal to explicit projection.

### 6.4 The text-share measure (E3)
For each item set, stream index *s*, definition *T* and k:
- Acc(T_k) = MaxSim accuracy when the REF and target tokens are first projected onto T_k.
- Acc(R_k) = mean MaxSim accuracy over **100 random k-dim subspaces** (fixed seeds).
- Also report the top-k PCA subspace of the image tokens themselves, and the residual (orthogonal complement) accuracy.
- Text share **S = (Acc(T_k) − Acc(R_k)) / (Acc(full) − 0.25)**. Not defined when Acc(full) − 0.25 < 0.05; reported as "not estimable".
- Primary statistic: S averaged over post-injection layers (Qwen s ≥ 3; Gemma and LLaVA have no DeepStack, so all s ≥ 0).
- Per-layer curves are always shown.
- Expect the authors' pattern for Qwen squiggles: probe accuracy is highest in early layers and falls toward chance in the middle layers (§4). There, S becomes noisy or "not estimable". That is a property of the data, to be reported, not hidden.

### 6.5 Known traps (check every time)
- **Massive activations:** a few huge dimensions can dominate cosine similarity. Report per-layer norms of image vs text tokens, and E3 with and without the top dimensions.
  - **Rule (fixed 2026-10-31, before data; our own rule, not from a paper):** per stream index, take the mean |value| of each dimension over the image tokens of the 260 basis items. A dimension is massive if this is more than 10× the median dimension's; at most the 16 largest are taken. The second view sets those dimensions to zero in every vector (regions and prompt) before any comparison; the bases stay the same.
- **Logit lens** undercounts text-likeness (LatentLens). Never use it as the only evidence.
- **Biased HSIC/CKA** looks high for random data at small n. If HSIC is used: unbiased estimator, ≥ 1,000 permutations, compared against random subspaces of the same size. Never compare a k-dim part against its (d−k)-dim complement directly.
- **Marker style:** keep it identical across conditions.
- **Repeated shapes:** the 8 named shapes repeat across items, so bootstrap by shape identity where needed.

---

## 7. Folder layout (create as steps progress; package name proposed, confirm with user)

```
ATML-Project2/
  CLAUDE.md, referance.md (project 1, read-only), VLMs language bias proposal.docx
  research_notes/      # literature check notes (2026-10-30)
  pyproject.toml       # `pip install -e .`, then `python -m nwf.<module>`
  kaggle/kernel/       # kernel-metadata.json, run.py (clones the repo at COMMIT, runs nwf.main --kaggle)
  kaggle/tests/        # CPU-only kernel infernalfishy/nwf-tests: runs the whole test suite on Kaggle under 5.16.1 then 4.57.1
  assets/              # small committed inputs: stimuli_manifest.json (hashes checked on Kaggle), captions.txt
  data/
    raw/               # authors' repo at the pinned commit, captions, KiloGram (untouched)
    stimuli/           # generated items: images + items.jsonl + manifest.json (hashes)
  src/nwf/
    main.py            # single entry point: local stages, or --kaggle [--smoke | --plan model:mode,...]
    paths.py
    config.py          # SINGLE SOURCE OF TRUTH: models + revisions, conditions, seeds, k values, layers
    shapes.py          # Step 2: draws every shape (named, outlines, icons, blobs, squiggles) on one canvas
    stimuli.py         # Step 2: lays items out with the authors' code (ported) + covariates + manifest
    models.py          # load model + processor at pinned revisions; full fp32 or the fast mode (§6.1)
    tokens.py          # box -> image-token positions (ports of the authors' functions)
    run_items.py       # bases stage + one forward pass per item with hooks + on-the-fly scoring + smoke checks
    maxsim.py          # MaxSim, inside/outside a subspace, SIEVE fit, all views of one stream index
    precision.py       # fp32 vs fast smoke comparison and verdict (§6.1)
    analysis/
      behaviour.py     # E1
      probe.py         # E2: MaxSim + linear probe
      text_share.py    # E3
      removal.py       # E4 (GPU, hook that projects out directions)
      figure.py
  tests/
  results/             # raw Kaggle outputs, NEVER edited (gitignored)
  analysis/            # small tables and figures (committed)
  reports/
```

---

## 8. The steps

Kaggle work happens only in the GPU steps (4 and 8, plus 7 if the hidden states must stay on Kaggle, see §6.2). Each GPU step is first run as a **smoke test** on 20 items per set.

### Step 0 — Setup and confirmations (no GPU) — done 2026-10-30
- [x] Evidence rules confirmed as trend-based grades (§1.3, user 2026-10-30).
- [x] Team: user alone for this study; Maryam is the project teammate.
- [x] Gemma access granted; local HF token reads it; official config checked (§3).
- [x] Package name `nwf` confirmed by the user (2026-10-30).
- [x] Gemma config and tokenizer loaded locally (transformers 4.57.1): vocab 262,208 (tokenizer length 262,145), embeddings tied. Both are library defaults, not in the config file, so re-check under Kaggle's transformers 5.
- [x] Letter tokens checked for both models (§6.1).
- [x] Kaggle CLI works (`kaggle quota`: 4.36 h GPU left on 2026-10-30, back to 30 h on 2026-10-31 00:00 UTC).
- [x] Library support: Qwen3-VL classes exist since transformers 4.57 and Gemma 3 since 4.50; Kaggle had 5.16.1 on 2026-10-29. Re-check the exact version in the Step 4 smoke run.
- **This computer cannot run the real models** (§10). Full-model loading is tested in the Step 4 smoke run instead.
- Deferred to Step 3: create the new private GitHub repo. **Ask the user first.**

### Step 1 — Read and reproduce the authors' pipeline (local, CPU) — done 2026-10-30
- [x] Sparse clone at the pinned commit in `data/raw/vnw_repo`; earlier-read files verified identical to it.
- [x] Resolved, all written into §4:
  - 8 vs 10 basic shapes (8).
  - Test-set sizes (paper 1,000; released 100).
  - How squiggles are drawn (thin open lines, no seed).
  - Gemma pan-and-scan (on).
  - The fine-tuning setups (Qwen full; Gemma language model only).
  - That the taught squiggles equal the test squiggles.
  - Token counts per item.
- [x] Found two confounds in the authors' contrast (filled vs line; look-alike shapes); §5 redesigned.
- [x] Found that the authors' own probe numbers disagree between files (74.2 vs 90.4); §4 rule: compare only within our own method.
- Still unknown (not blocking): what `squiggles_30_TRAIN` contained for the task-fine-tuned models (matters only for RQ3).

### Step 2 — Build stimuli (local) — done 2026-10-31
- **Result:** `python -X utf8 -m nwf.stimuli` built 13 sets into `data/stimuli` (11 × 600 items, exactly 150 per letter; the 2 `rep_` sets as released, 100 items each; all pictures 800×600). Checked a second way by an independent script (not in the repo):
  - every hash in `manifest.json` recomputed and matching; no duplicate ids; calibration flags and pool halves correct;
  - answer key checked from the pixels: in every item the REF shape matches the answer shape better than any other (all 6,600 generated items, and the 159 `rep_` items with no box past the edge; the smallest gap to the closest wrong shape is 0.11);
  - the red labels sit under the right shapes in every item;
  - the authors' shape files copied pixel-identical;
  - two sets (`unnamed_line`, `named_cx`) rebuilt from scratch gave byte-identical manifests;
  - by eye, one item per set with boxes drawn: correct.
  - A first run of the pixel check flagged 104 items; the cause was the check itself (it took dark red shapes for red label text). With the authors' strict red rule the check passes on every item.
- Generator per §5, reusing the authors' rendering code (layout, colours, labels, box computation) and squiggle recipe, but **with fixed seeds**. Write `items.jsonl` (id, set, shapes, answer, boxes, covariates, seed) plus a manifest with hashes.
- Model naming check for `named_fill`, `named_line` and `named_cx` (single image: "What shape is this? Answer with one word."), run in Step 4's smoke run; shapes that fail are moved out of the named sets before the main run.
- **Check by eye:** 20 items per set rendered correctly, boxes drawn over the images line up with the shapes, letters balanced.
- **Done when:** counts match §5.1, no duplicate ids, the box overlay check passes.

### Step 3 — Pipeline code + CPU tests (local) — done 2026-10-31
- Tests with known answers:
  - Box→token mapping on a synthetic image with a known grid.
  - Hooks equal `output_hidden_states` where expected. Show and document the DeepStack difference.
  - Single-token letters after `(`.
  - Projection maths (a vector in the subspace stays, an orthogonal vector vanishes).
  - MaxSim matches the authors' function on random tensors.
  - S equals 0 when T is itself a random subspace (in expectation).
- Measure tokens per item and fix the storage plan (§6.2).
- **Done when:** all tests pass under the local mirror of Kaggle's library versions.
- **Progress (2026-10-31, end of session):**
  - [x] All code written (§7) and pushed: commit `0dd13fd` (pipeline + tests), `784ef03` (test kernel).
  - [x] Passed locally under **both** 4.57.1 and 5.16.1: `test_tokens` (7), `test_maxsim` (7), `test_letters` (2), `test_hooks` (5). Passed under 4.57.1 only so far: `test_removal` (2), `test_run_items::test_smoke_pipeline_on_tiny_models[qwen]`.
  - [ ] **Not yet seen passing:** the rest of `test_run_items`, `test_main` (full smoke orchestration with the real settings), and the whole suite in one go. The laptop is too slow (over 4 h for both versions), so the suite was moved to Kaggle.
  - **Running when the session ended:** Kaggle CPU kernel `infernalfishy/nwf-tests`, version 2, started by the user from the editor on 2026-10-31 about 02:40 PKT (secrets `HF_TOKEN` and `GITHUB_TOKEN` attached; the log confirmed both repos were cloned at the pinned commits; no GPU). It runs the full suite under the preinstalled transformers (expected 5.16.1), then under 4.57.1, and writes `tests_preinstalled.log`, `tests_4.57.1.log`, the JUnit XML files and `tests_summary.json`.
  - **Result of that run (read 2026-10-31, `results/tests_kaggle_01`):** 21 passed and 20 failed, the same under 5.16.1 and 4.57.1 (about 3 min each).
    - All 20 failures had **one cause**: the folder `/kaggle/input/nwf-stimuli` did not exist, although the kernel metadata lists the dataset as attached. 18 failed with `FileNotFoundError`; the 2 in `test_main` failed the stimuli hash check, because an empty folder has the wrong hash. No test failed on its own logic.
    - The 21 tests that need no stimuli passed under both versions on Kaggle (stimuli rendering, tokenizers, hooks setup, maxsim and others).
    - **Fix (2026-10-31, commit `d96d7dd`, pushed; test kernel pushed as version 3 from the command line, so it gets no secrets and fails):** both Kaggle scripts now find the stimuli by searching `/kaggle/input` for `rep_named/items.jsonl`, stop unless exactly one is found, and print the path. `kaggle/kernel/run.py` had the same bug and would have failed the smoke run; its resume folder is now found by its `environment_main.json`. The old fixed path would have made a resumed main run silently start from zero. Tested locally on fake folder layouts: old path, nested path, none, two copies, the real `data/stimuli`.
    - The code under test (`0dd13fd`) is unchanged, so `COMMIT` in `run_tests.py` stays.
  - **Kaggle test run 2 (version 4, started by the user from the editor 2026-10-31 08:34 UTC, `results/tests_kaggle_02`): all 41 tests passed under both 5.16.1 and 4.57.1.**
    - Checked from the JUnit XML: 41 cases, 0 failures, 0 errors, 0 skipped in each version, and the same 41 test ids as run 1.
    - Run time: 38.6 min (5.16.1), 25.7 min (4.57.1), 1.1 h in total.
    - The stimuli were found at `/kaggle/input/datasets/infernalfishy/nwf-stimuli`. `test_main` ran the stimuli hash check on that real dataset, so **all 13 uploaded sets match the local build on Kaggle**.
    - `test_main::test_smoke_run_end_to_end_with_real_settings` took 1,800.2 s under 5.16.1 but 1,018 s under 4.57.1. There is no timeout or wait in the code; the test asserts that all 4 steps ended with code 0 inside its 1 h deadline and that every output exists, so nothing was cut short. Likely cause: the first run read the dataset files cold **[unverified]**.
    - `src`, `tests`, `assets` and `pyproject.toml` are identical in `0dd13fd` (tested) and `d96d7dd` (latest; only the Kaggle scripts changed).
  - **Reading a test run:** `kaggle kernels status infernalfishy/nwf-tests`, then `kaggle kernels output infernalfishy/nwf-tests -p results/tests_kaggle_NN`. Read `tests_summary.json` and both logs. Fix real bugs, rerun the affected tests locally, commit, and start a new version from the editor (CLI-pushed versions get no secrets). Never change what a test checks just to make it pass; ask the user about anything ambiguous. When everything passes under both versions, mark Step 3 done and ask the user before the Step 4 smoke run.
  - The GitHub token in the Kaggle secret `GITHUB_TOKEN` can read `no-word-for-it` (confirmed by the clone).

### Step 4 — Kaggle: smoke run, then the main run (GPU)
- Smoke (`MODE="smoke"` in `kaggle/kernel/run.py`, plan `main.SMOKE_PLAN`): 20 items per set on `qwen`, `qwen_names` and `gemma`, each run twice (fully fp32 and the fast mode) for the precision check in §6.1. Each run fits its own bases on 200 captions and 2 items per set. The fp32 runs also save the authors-style 5-token generated answers (first 5 items per set) and the naming check answers. Estimated about 2 GPU h (mostly Gemma fp32).
- **Check:**
  - No NaN.
  - The §6.1 precision check (`results/precision_{model}.json`) decides each model's dtype before the main run.
  - Letter-logit answers agree with the authors' generate-and-parse on the 65 generated items.
  - Replication sets land near the paper's values.
  - Time per item, which gives the full-run estimate.
- **Progress (2026-10-31):** `COMMIT` in `run.py` set to `d96d7dd` (commit `1e03e72`, pushed). GPU kernel `infernalfishy/nwf-effect-study` created by a command-line push (version 1). It failed as expected at `get_secret("HF_TOKEN")` (HTTP 400, no secrets); the stimuli search before it ran fine on the GPU image. GPU quota then: 30 h left, resets 2026-11-07. The user then started version 2 from the editor (11:17 UTC).
- **Smoke run result (version 2, `results/kaggle_smoke_01`, read 2026-10-31):**
  - All 8 steps ended with code 0 in 4.10 h. All 13 stimulus sets matched their hashes on Kaggle. Image: 2× T4, torch 2.11.0+cu128, transformers 5.16.1, Python 3.13.15.
  - **Precision check:** `qwen` passes (cos min 0.99901, 99.2% same letter, 2 flips, both near-ties) → fp16. `qwen_names` passes (0.99909, 100%) → fp16. **`gemma` fails** (0.99906 but 97.7% same letter, 6 flips, not all near-ties) → **fully fp32** under the §6.1 rule.
  - Letter argmax equals the authors-style generated answer on 65/65 items for each of `qwen`, `qwen_names` and `gemma` (fp32).
  - Item ids match the first 20 ids of every set in `data/stimuli`; letter shares: base Qwen picks A on only 20 of 260 items (the known A bias).
  - **Naming check** (whole-word match against `config.NAMES`): `qwen` 25/26, misses only `teardrop` (answered "teardrop", while the accepted word is "drop"); `gemma` 25/26, misses only `club` (answered "Clube"). With substring matching both would be 26/26. **The matching rule is not fixed yet; ask the user.** `qwen_names` calls circle "anchor", oval "blade" and square "shield": it applies its taught squiggle names to basic shapes.
  - **Likely cause of the Gemma gap (read 2026-10-31 in the library source):** the authors run Gemma with `attn_implementation="flash_attention_2"`, bf16, padded batches. In transformers 4.57.1 **and** 5.16.1, `masking_utils.flash_attention_mask` ignores the `mask_function`, so Gemma 3's two-way attention among the tokens of one image (`token_type_ids_mask_function`, an `or_mask_function`) is dropped and image tokens attend one way only, as in text. Our `sdpa` path keeps the two-way image attention Gemma 3 was trained with. Which transformers version the authors used is **[unverified]**. Their behaviour file names also do not record pan-and-scan (only their probe files say `psTrue`), and no Gemma behaviour files were released. Possible check next week (about 1 GPU h, needs a small code change, ask the user): Gemma on the two `rep_` sets with one-way image attention.
  - **Gemma replication looks off (20 items only):** `rep_named` 0.90 and `rep_squig30` 0.50 vs the paper's 50.6 and 30.5. Qwen is near the paper: 0.65 / 0.25 vs 54.1 / 29.0. Investigate before going on (Step 5 rule); the main run has all 100 released items.
  - **Gemma probe near-ties:** in the raw view the top-2 MaxSim scores differ by a median of only 0.01–0.2% (relative) from stream 5 on; with massive dimensions removed the gap is 4–13% (Qwen: 2–7% in both views). So the raw-view Gemma probe is dominated by massive activations, which also explains its falling fp32 vs fast probe agreement (to about 86%). The massive-removed view is the meaningful one for Gemma.
  - **Time per item** (chunks of 20, bases excluded): Qwen fp32 3.9 s, fp16 2.5 s; `qwen_names` fp16 2.4 s; Gemma fp32 17.4 s (fast 8.9 s). Bases: Qwen about 1 min, Gemma fp32 about 8 min. Removal (fast) 2.1 min for 180 passes.
  - **Main-run estimate from this:** 6,800 items each: `qwen` fp16 about 4.8 h, `qwen_names` fp16 about 4.6 h, `gemma` fp32 about 33 h, removal about 1.5 h; about 44 h in total, more than the 30 h left this week. Needs a user decision.
- Main run: all sets on `qwen`, `gemma` and `qwen_names` (plus `gemma_names` if time allows). Raw outputs go to `results/kaggle_run_01/`, read-only.
- **Done when:** the manifest shows every set × model finished.

### Step 5 — E1: behaviour gap (local)
- Accuracy per set with CIs; the E1(b) regression.
- Replication check on the released 100-item sets (expect wide CIs, about ±9 points): Qwen squiggles-30 should land near the authors' released 32% (paper, 1,000 items: 29.0); basic shapes near the paper's 54.1. Gemma: paper 50.6 / 30.5.
  - A big difference is investigated (rendering, dtype, scoring) **before** going on, and logged.

### Step 6 — E2: encoded but unused (local)
- MaxSim per layer per set. Fix the probe layer on the calibration split, then E2(a), (b), (c).
- Linear probe (held-out shapes, Hewitt–Liang control task) as a robustness column.

### Step 7 — E3: where the information sits (local, CPU or Kaggle if storage needs it)
- T1/T2/T3 × k × layer; random and PCA baselines; S curves and primary S.
- Massive-activation rule applied. Optional unbiased-HSIC column.
- E5 on `qwen_names` (and `gemma_names`).

### Step 8 — E4: removal check (GPU, small) — built in Step 3 and run inside the main Kaggle run (user decision 2026-10-31: Qwen only)
- A hook removes the top-k text directions (T2, k ∈ {2, 16}) from image tokens at every post-injection layer. Measure letter accuracy per set.
- Control: remove random directions with matched projected norm.
- **As built (`removal.py`, fixed 2026-10-31 before data):**
  - Pre-hooks on decoder layers s ≥ 3 (Qwen) replace every image token x by x − UUᵀx (computed in fp32), with U the top-k caption directions at that stream index.
  - Random control: a random subspace of size m_s = round(f_s · d), clipped to [k, d], where f_s is the share of image-token energy that the top-k caption directions hold at stream s (measured on the 260 basis items in the bases stage). A random m-dim subspace holds m/d of any vector's energy on average, so it removes the same energy on average. 3 seeded draws per k (seed `[0, 1000 + 100k + draw]`).
  - Items 100–299 (200 test items) of `named_fill`, `named_line`, `unnamed_fill`, `unnamed_line`; every item scored with no removal plus the 8 removals (about 7,200 forward passes, about 1 GPU h). Results in `results/{model}_{mode}/removal/`.
  - Tested: the hook leaves layers 0–2 and the text tokens untouched and leaves no trace of U in the image tokens from layer 3 on; "no removal" equals the main run.

### Step 9 — Write-up and decision (local)
- `reports/effect_study_summary.md`: the figure, which of E1–E5 hold and how strongly, the replication check, limitations (fp16/fp32 vs bf16; identical-copy task; MaxSim is not a trained probe; one task family), and the go / no-go per §1.3.

---

## 9. Rules for working in this repo

- §0 always applies.
- **No training in this study.** Everything uses released checkpoints. If a step seems to need training, stop and ask.
- **Pin everything:** model revisions (§3), the authors' repo commit, seeds, library versions. Log the resolved commit sha of every model on every run.
- `results/kaggle_run_*` is read-only raw data.
- Kaggle code must be resumable: one model per subprocess, atomic writes, `.done.json` markers, deadline checks (same pattern as project 1).
- Windows shell: absolute paths; run Python as `python -X utf8`.

### Code style rules (carried over from project 1; the user wants really easy code every time)
**General**
- Code should be really easy to read and look plausibly human written, not AI-generated boilerplate.
- Prefer plain, simple solutions over clever ones.
- No space after commas and no space after colons in dict literals, e.g. `f(a,b,c)`, `{"a":1,"b":2}`. This is deliberate; do not "fix" it.
- Named colors only in plots (`"steelblue"`, `"darkorange"`, `"firebrick"`, `"seagreen"`), never hex codes.
- No `try`/`except` anywhere. Let errors surface.
- Minimal prints: one summary line per meaningful step, no verbose logging.
- Only define a function when it is reused more than once. No helper functions for one-off code.
- All imports at the very top of the file, one per line: stdlib first, then third-party, then torch/sklearn. No inline imports later.

**Comments**
- Sparse and lowercase.
- Only where the reasoning is not obvious from the code itself.
- Action comments use the `-ing` form: `# dropping the identity add so out is just the conv path`.
- No dashes, commas or forward slashes in comments. Write the sentence out in full instead.
- Never end a comment with a period.
- Never mention an assignment, a paper, an author or a file path in a comment.

**Docstrings**
- Every function gets a short docstring in triple double quotes.
- Plain simple English describing what the function does.
- Same character rules as comments: no dash, comma or slash, and no period at the end of any line.

**Project structure**
- Every script is runnable on its own with `python -m nwf.<module>`.
- One entry point runs every stage in order: `python -m nwf.main` (local) and `python -m nwf.main --kaggle` (GPU stages).
- Heavy files go in gitignored folders and are never committed: `data/`, `results/`, `cache/`.
- Only small result tables and figures get committed (`analysis/`, `reports/`).

**Editing existing code**
- Never change preexisting code, comments or formatting you were not asked to change.
- When fixing something, touch only the exact lines needed.

**Git**
- Nothing in the repo may reference Claude or an AI assistant: code, comments, README, commit messages. **Exception (user decision 2026-11-01):** `CLAUDE.md` itself is committed to `main` so teammates' Claude Code gets the full context; keep it free of secrets. `.claude/`, `referance.md`, `research_notes/` and the proposal .docx stay excluded through `.git/info/exclude`, never `.gitignore`, so the links to them in this file only work on the user's machine.
- The repo is private. Commit as the user's git identity (Mohid Ali).
- No Co-Authored-By trailer on commits.
- Short, plain, human-sounding commit messages.

---

## 10. Machines

### This computer (checked 2026-10-30)
- Intel i7-1355U, **7.5 GB RAM (often under 1 GB free)**, no GPU. C: has about 13 GB free; **D: has about 130 GB free**. The Hugging Face cache already lives on D: (`D:\ATML_PA1_cache\huggingface`).
- **Too small to load the real models** (Qwen3-VL-2B needs about 4.3 GB in bf16 and 9 GB in fp32). Local tests use tiny random stand-in models built from the real configs, plus the real tokenizers and processors (as in project 1).
- Python 3.11.9 with transformers 4.57.1, torch 2.12 (CPU), huggingface_hub 0.36.2, pytest 9.2. `qwen_vl_utils` is not installed and not needed (images go straight to the processor).
- TensorFlow is installed and gets imported by transformers (harmless log lines).
- The project folder is on OneDrive; keep big files (model weights, hidden states) out of it.
- **Local mirror of Kaggle's libraries (built 2026-10-31):** `D:\ATML_P2_venv` (Python 3.11) with transformers 5.16.1, torch 2.11.0+cpu, torchvision 0.26.0, tokenizers 0.23.1, accelerate 1.14.0, safetensors 0.8.0, huggingface_hub 1.29.0; `nwf` installed editable. Every test must pass in **both** it and the system Python (4.57.1).
- **Running the tests:** set `HF_HOME=D:\ATML_PA1_cache\huggingface` and `HF_HUB_OFFLINE=1` (processors and configs at the pinned revisions are cached there), then `python -X utf8 -m pytest tests -p no:warnings` or the same with `/d/ATML_P2_venv/Scripts/python.exe`. The full suite takes over an hour on this laptop (tiny stand-in models on CPU, little free RAM). Never kill all python processes; stop only the test's own process id.

### Kaggle lessons from project 1 (verified then; re-check what matters)
- Account `infernalfishy`; secrets `HF_TOKEN` and `GITHUB_TOKEN`.
- **Secrets reach only versions started with Save Version in the browser editor.** CLI-pushed versions fail with HTTP 400 on `get_secret`.
- Hardware: 2× T4 (15 GB each, compute capability 7.5, **no bf16, no flash-attention 2**), 31 GB RAM, `/kaggle/working` 20 GB (saved), large free space under `/tmp` (not saved).
- 2026-10-29 image: Python 3.13, transformers 5.16.1, torch 2.11 (+cu128). Pin with `docker_image_pinning_type: original`.
- **CPU image measured in the test run (2026-10-30, image `gcr.io/kaggle-images/python@sha256:17b2fc7f…`):**
  - Python 3.13.15, torch 2.11.0+cpu, torchvision 0.26.0+cpu, transformers 5.16.1, tokenizers 0.23.1, accelerate 1.14.0, safetensors 0.8.0, huggingface_hub 1.29.0.
  - numpy 2.1.3, scipy 1.16.3, pillow 13.0.0, matplotlib 3.10.0, pandas 2.3.3, pyarrow 23.0.1.
  - The ML libraries match the local mirror `D:\ATML_P2_venv`.
  - Installing transformers 4.57.1 there brings tokenizers 0.22.2 and huggingface_hub 0.36.2. pip warns that gradio and diffusers conflict with them; we use neither.
  - The GPU image may differ (torch +cu128); record it from the smoke run's `environment_smoke.json`.
- **Attached datasets are mounted at `/kaggle/input/datasets/<owner>/<slug>`**, not `/kaggle/input/<slug>` (seen in the test run log, 2026-10-31). The Kaggle scripts search `/kaggle/input` instead of using a fixed path, so either layout works.
- transformers 5 changed sentencepiece-only tokenizers (lost leading `▁`). All three generator models here ship a `tokenizer.json`, but check token ids for the letters and the pre-fill anyway.
- Code reaches Kaggle by cloning the private repo at a pinned commit; prompt/stimulus files are hash-checked after rebuild.
- GPU quota is about 30 h/week; check with `kaggle quota`.
- Kaggle CLI 2.2.5 logs in with OAuth (`~/.kaggle/credentials.json`). On 2026-10-31 the saved login stopped working ("Authentication required") while `kaggle auth login` still said "already logged in". Fix: the user runs `kaggle auth login --force` in a terminal. Claude never handles the token.
- **Stimuli dataset:** `infernalfishy/nwf-stimuli` (private), uploaded 2026-10-31 from a staged copy that passed the hash check. The dataset's file list shows the set folders at its top level (`<set>/...`, plus `manifest.json`), and spot-checked sizes match. Inside a kernel it is at `/kaggle/input/datasets/infernalfishy/nwf-stimuli`, and all 13 sets matched the local hashes there (test run 2, 2026-10-31).

### Kaggle limits and GPU budget (checked 2026-10-30)
- Kaggle docs (kaggle.com/docs/notebooks, read 2026-10-30): **12 h maximum per session** (GPU), `/kaggle/working` 20 GB, T4 ×2 machine has 4 CPU cores and **29 GB RAM** (not 31). Linking a Colab Pro / Pro+ account adds 15 / 30 extra GPU h per week (experimental promotion).
- Kaggle Terms of Use (version 22 Jun 2025, read 2026-10-30): one person may not have or operate **more than one active Kaggle account**; the penalty can be a ban. Sharing an account or password is also not allowed. **Never make a second account.** More GPU time must come from the weekly reset, Maryam running part of the work from her own account, or Colab Pro.
- Rough compute per item (from the pinned configs: 2 × weights × tokens plus attention): Qwen about 6 TFLOP, Gemma with pan-and-scan about 44 TFLOP (7×, mostly its vision encoder on 6 crops of 896 px). With T4 at about 20 TFLOPS effective in fp16 and 5 in fp32, plus overheads: Qwen fp16 about 1–2 h for all 6,800 items, Gemma fp32 about 17–21 h (does not fit one 12 h session). With Gemma's vision tower in fp16 (only if the §6.1 check passes) about 9–10 h. Whole plan (smoke, qwen, qwen_names, gemma, gemma_names on the E5 sets, E3 on Kaggle, E4 on Qwen only): about 27–37 h with Gemma fully fp32, about 17–24 h with its vision tower in fp16. **These are estimates; the smoke run measures the real time per item.**

---

## 11. Decisions and deviations log

| Date | Decision | Reason |
|---|---|---|
| 2026-10-30 | Project 2 = "No Word for It" effect study; project 1 closed | project 1 found the opposite effect (see referance.md §9) |
| 2026-10-30 | Effect study before any mechanism work; RQ2 / RQ3 gated on §1.3 | professor's requirement; lesson from project 1 |
| 2026-10-30 | Evidence rules E1–E5 drafted with CIs, no fixed cut-offs | same approach the user chose in project 1 |
| 2026-10-30 | Evidence judged as trend grades (clear / moderate / weak / absent / reversed); a moderate, consistent effect is enough to go; plan numbers are defaults, not bars. Kept strict: main views fixed before data, every setting shown | user decision, before any data: the rules were drafted by Claude, not taken from the literature |
| 2026-10-30 | Initial effect study run by the user alone; Maryam is the project teammate | user decision |
| 2026-10-30 | Gemma access confirmed; official config read at the pinned revision | HF page shows access granted; `auth_check` passes for the local token |
| 2026-10-30 | Authors' repo kept as a sparse clone (shapes, name teaching, keypoint code only) | full clone is about 500 MB, mostly face photos we do not use; the download was very slow |
| 2026-10-30 | Core stimulus design changed to a 2×2 (named or unnamed × filled or line) plus a look-alike covariate; E1(b) updated to match. Made before any data | Step 1: the authors' named shapes are filled and distinct, their squiggles are thin lines that look alike, so "name" was mixed up with "fill" and "look-alike" |
| 2026-10-30 | E5 uses the authors' taught squiggles vs their untaught `squiggles_30_v2` | the taught squiggles are byte-identical to the test squiggles (sha256), so taught vs untaught is a clean contrast |
| 2026-10-30 | Gemma run with pan-and-scan on; `use_fast` set explicitly | matches the authors' saved Gemma results (`psTrue`); the processor default changes across transformers versions |
| 2026-10-30 | Our probe numbers are compared only across our own conditions, never to the authors' absolute numbers | the authors' saved files give 74.2 and 90.4 for the same model and test type |
| 2026-10-30 | Private repo `Mohidali2005/no-word-for-it` created (user approved); CLAUDE.md, referance.md, research_notes/ and the proposal .docx are kept out through `.git/info/exclude`; `core.autocrlf=false` in this repo | same setup as project 1; nothing in the repo may mention Claude |
| 2026-10-30 | All non-authors shapes redrawn by one renderer (200 px canvas, 4 px line, longer side 180 px); `named_fill` uses our redraw of the authors' 8 shapes, not their PNGs | the 2×2 cells must differ only in name and fill; the authors' PNGs stay in `rep_named` |
| 2026-10-30 | Item layout = authors' code ported line by line, with byte-identical output verified against their own functions; shape centres kept 0.3 (not 0.25) from the sides | at 0.25 shapes can be cut off at the picture edge, which the authors' released items show (44 of 1,600 boxes) |
| 2026-10-30 | Look-alike covariate changed from "mean pairwise mask overlap" to "highest blurred-mask correlation between the REF shape and a wrong answer"; signatures padded with a 50% margin | seen in the 20-item preview, before any model data: overlap of thin lines is near 0 for any pair, and a tightly cropped filled square gave an undefined value; the closest distractor is what makes a choice hard |
| 2026-10-30 | Stimulus pools of 40 split into halves by item parity | lets a linear probe be tested on shapes it never saw |
| 2026-10-30 | Hidden states captured with our own pre-hooks, not `output_hidden_states` | HF records Qwen3-VL states before DeepStack injection (verified in code) |
| 2026-10-30 | Calibration split (first 100 items per set) chooses the probe layer | avoids picking the best layer on the same data that is tested |
| 2026-10-30 | Unnamed-simple and named-complex conditions added | the authors' named vs squiggle contrast mixes "no name" with "complex" |
| 2026-10-30 | Use the authors' released checkpoints instead of training | they released name-taught and task-fine-tuned models for both Qwen and Gemma |
| 2026-10-30 | Direct answers only (no chain-of-thought) in this study | CoT needs an 8B extractor model and long generation; direct answers carry the published gap (Qwen 54.1 vs 29.0) |
| 2026-10-31 | dtype rule changed before any data: Gemma's vision tower may run in fp16 (projector and language model stay fp32), and Qwen's fp16 is now judged by a fp32 comparison, not only a NaN check. Both decided by the §6.1 precision check (no NaN or inf; image-token cosine ≥ 0.999; ≥ 98% same letters with every flip a near-tie); otherwise fully fp32 | user approved: Gemma fully fp32 needs about 17–21 GPU h, more than one 12 h session; fp16 is finer than the authors' bf16, and the same setting for every condition cannot create a named vs unnamed gap |
| 2026-10-31 | Never use a second Kaggle account | Kaggle Terms of Use forbid more than one account per person; extra GPU hours come from the weekly reset, Maryam's own account, or Colab Pro |
| 2026-10-31 | Probe scores computed on the GPU right after each forward pass; hidden states never stored (§6.2) | about 100 GB of states per model cannot leave Kaggle; scoring on the fly keeps the run resumable across sessions and saves about 0.7–1 MB per item |
| 2026-10-31 | Subspace implementation details, the massive-dimension rule, image-PCA from the first 20 items of each set, T3's matched random baseline (§6.3, §6.5) | the plan left these to Step 3; fixed before any model data |
| 2026-10-31 | Naming check: the main run scores all items; at analysis, items holding a shape the model failed to name are excluded from that model's named sets (reported) | same result as removing them before the run, but the main run no longer has to wait for the naming check |
| 2026-10-31 | Authors-style generation check on the first 5 items per set (65 items), fp32 run only | generating with Gemma in fp32 is slow; 65 items are enough to confirm that the letter argmax matches the parsed answer |
| 2026-10-31 | Precision check also run for `qwen_names` | it is a different set of weights, so its activations may differ in range |
| 2026-10-31 | Image processors: Qwen fast, Gemma slow, as each repo declares | the authors used defaults of an unknown version; each repo's own declaration is the clearest choice, and pixels are identical across library versions |
| 2026-10-31 | Captions from `yerevann/coco-karpathy` (pinned commit, 1 MB) | user decision; the standard COCO validation captions in a small pinned file |
| 2026-10-31 | Stimuli reach Kaggle as the private dataset `infernalfishy/nwf-stimuli`; every run checks all file hashes against `assets/stimuli_manifest.json` before starting | user decision; rebuilding on Kaggle may not be byte-identical (different matplotlib) and would cost GPU time |
| 2026-10-31 | E4 built in Step 3 and run in the main Kaggle run, Qwen only; main view k = 16 on the four core sets (§1.3) | user decision: all GPU work in one go; Gemma would cost about 10× more for a supporting check |
| 2026-10-31 | Random subspaces stored side by side and scored with one matrix product per stream index | same answers (tested), several GPU hours less on a T4 |
| 2026-10-31 | Gemma runs fully in fp32; `qwen` and `qwen_names` in fp16 | the §6.1 precision check in the smoke run (rule fixed before data): Gemma 97.7% same letter with flips that are not near-ties |
| 2026-10-31 | **Deviation, made after seeing the smoke naming answers:** a shape counts as named if any accepted word in `config.NAMES` appears **inside** the model's lowercased answer (substring), not only as a whole word | the accepted list has "drop" and not "teardrop", which only makes sense with substring matching; the rule had not been written down before the smoke run. Seen: whole-word matching fails "teardrop" (Qwen) and "Clube" (Gemma), substring passes both. User delegated the choice to Claude |
| 2026-10-31 | **Gemma reduced (before any Gemma main data, after its smoke run):** fp32 on the 4 core 2×2 sets (first 300 items each: 100 calibration + 200 test) and the 2 `rep_` sets (all 100), plan entry `gemma:full:named_fill+named_line+unnamed_fill+unnamed_line+rep_named+rep_squig30:300`; about 6.8 h of items + about 1.3 h of bases (bases still use the first 20 items of all 13 sets, as fixed in §6.3). Dropped for Gemma: complexity sweep, `named_cx`, `taught30` / `untaught30`. New `--items` option in `run_items`, fourth plan field in `main`, test `test_main::test_a_plan_entry_keeps_only_its_sets_and_first_items` (passed locally under 4.57.1 and 5.16.1 together with the deadline test; commit `a7527aa`). The Gemma run must pin `COMMIT` to `a7527aa` or later. Because the first 300 items are only roughly letter-balanced (e.g. `unnamed_line` test answers A 48, B 43, C 47, D 62), Gemma's accuracy is also reported as the mean of the four per-letter accuracies | full Gemma fp32 measured at about 33 h; its role in the go / no-go is only to point the same way as Qwen on E1–E3, which the core sets carry; 200 test items give about ±7 points. User approved |
| 2026-10-31 | Main run split by model: `qwen` fp16 (plus removal) and `qwen_names` fp16 now (about 11 h), Gemma fp32 (about 33 h) from the next weekly quota | the whole plan needs about 44 GPU h, 30 h were left; Qwen decides the go / no-go; the Gemma replication gap (smoke: 0.90 / 0.50 vs paper 50.6 / 30.5) is investigated first. User delegated the choice to Claude |
| 2026-11-01 | E1 analysis choices (Claude, written after the quick raw accuracies had been seen, but before the E1 script was run; to be confirmed by the user): E1(a) main view = `named_fill` minus `unnamed_line`, with `rep_named` minus `rep_squig30` (all 100 items) as the replication; bootstrap and regression clusters = REF shape identity | the `rep_` sets are all calibration items, so they have no test items; `named_fill` vs `unnamed_line` is the redrawn version of the authors' contrast (named basic shapes vs squiggles n=30) |
| 2026-11-01 | **Gemma analysis choices (Claude, written before any Gemma main-run data was read; to confirm with the user):** main probe view for Gemma = **massive-removed** (raw view shown next to it); all other E1–E4 choices as for Qwen (test items 100–299, `rep_` sets whole, clusters = REF shape, the same set pairs, post-injection streams = all s ≥ 0); accuracy also reported letter-balanced (§11 Gemma row) | the smoke run (first 20 items per set, all inside the calibration split) showed raw-view top-2 MaxSim gaps of 0.01–0.2% from stream 5 on, i.e. near-ties set by massive dimensions; the massive-removed view gave 4–13% gaps |
| 2026-11-01 | `CLAUDE.md` committed to `main` (removed from `.git/info/exclude`) together with the E1–E4 analysis code and result tables; the PDF, `referance.md` and `research_notes/` stay local | user decision: teammates' Claude Code should have the full project context |
| 2026-11-01 | **Exploratory lead opened (user decision, after seeing E3/E4):** "loud" high-variance image-token directions may drown out fine shape detail (removing the top 64 image-PCA directions raises the squiggle probe 0.74 → 0.98). Step 1 (CPU, stored data): what those directions encode (colour, position, shape). Step 2 (GPU, about 1 h, ask first): remove them inside the model and see whether answers improve. Everything found on the existing data is labelled **exploratory**; a new hypothesis and its pass/fail rule are written down here **before** step 2 and judged only on the new run | user: a result against H can open a new direction; testing a new idea on the data that suggested it would be circular |
| 2026-11-01 | E2 analysis choices (Claude, before the E2 script was run; to be confirmed by the user): main probe view = raw vectors (massive-removed view shown next to it, with its own probe layer chosen by the same calibration rule); probe layer pooled over the calibration items of all 13 sets (the `rep_` sets included, since all their items are calibration items); E2(b)/(c) use the same set pairs and clusters as E1 | follows §6.2 and §6.5; keeps E1 and E2 comparable |

---

## 12. Status

**Where we are (2026-11-01, late evening):** Steps 0–3 done. Qwen and Gemma main runs done and downloaded. E1–E4 analysed on `qwen`; E1–E3 analysed on `gemma` (details below; E4 is Qwen only).
- **Headline (Qwen):** E1 Clear, E2 Clear (E2(c) Absent), **E3 not supported (T2 reversed, T1/T3 not text-specific), E4 Absent at the main view**. Under §1.3 → **stop or pivot**; not yet discussed with Maryam and the professor.
- **Headline (Gemma):** points the same way as Qwen. E1 Clear, E2 Clear (E2(c) Moderate here, unlike Qwen), **E3 not supported (T2 reversed again)**. So the two models agree, and the §1.3 decision stays **stop or pivot**.
- **Update PDF for the group:** `reports/effect_study_update_2026-11-01.pdf` (4 pages, plain language, 2 figures; built by a scratchpad script, not committed). Shares the results above and the decision to make.
- **Exploratory lead being followed (user decision, §11):** "loud" top image-PCA directions drown out fine shape detail. Step 1 (CPU, stored pooled region vectors: can colour, position and shape identity be read from the top 64 image-PCA directions vs 64 random directions vs the rest?) was running at the end of the session (scratchpad script `loud_step1.py`, not in the repo; slow on the laptop). First row (named_fill, s = 3): top 64 hold 70% of the energy; colour R² top 0.44 / random 0.40 / rest 0.91; position 0.96 / 0.93 / 0.99; shape identity 1.00 everywhere (uninformative for named). Squiggle rows are the informative ones. **Next:** read step 1, write the new hypothesis and its pass/fail rule into §11, then ask the user before the step 2 GPU run (remove the loud directions inside the model, likely gently, e.g. one early layer or the per-image shared part, since every E4 cut broke the model).
- **Qwen main run (version 4), done and downloaded:** finished after 10.77 h, all steps exit code 0 (`qwen:fast` 271.5 min, `qwen:fast:removal` 76.2 min, `qwen_names:fast` 275.8 min). Output in `results/kaggle_run_01/results` (8.6 GB, read-only; inside OneDrive but git-ignored, user was told). Checked: 6 chunks of 100 per generated set for both models, 1 chunk per `rep_` set, removal 2 chunks (items 100–299) in each of the 4 core sets, no non-finite value in any npz.
- **Gemma main run (started by the user 2026-11-01 from the editor), done and downloaded:** commit `a7527aa`, plan `gemma:full:named_fill+named_line+unnamed_fill+unnamed_line+rep_named+rep_squig30:300`, fp32, transformers 5.16.1. Finished after 7.97 h, exit code 0 (bases 73 min). Output in `results/kaggle_run_02/results` (1.5 GB, read-only). The first download stopped partway (one truncated npz, `unnamed_line` missing); it was downloaded again in full with `-o`. Checked: 3 chunks of 100 per core set and 1 per `rep_` set, item ids equal to the first 300 (100) ids of each set, no non-finite value in any npz or in `bases.npz`, all 13 stimulus sets matched their hashes on Kaggle.
- **Gemma E1–E3 (`--run kaggle_run_02/results --name gemma_full`; outputs `analysis/e{1,2,3}_gemma_full.json`):** test items 100–299 of the core sets, `rep_` sets whole, same clusters and pairs as Qwen, post-injection streams = all 35 (s ≥ 0). The analysis scripts now skip sets the run does not have. Every E1/E2 number below and two E3 shares recomputed by independent loop scripts (scratchpad, not in the repo): identical. Naming check: Gemma passes 26/26 with substring matching (§11), so no item is excluded.
  - **E1:** accuracies named_fill 0.835, named_line 0.915, unnamed_fill 0.390, unnamed_line 0.350 (letter-balanced 0.839 / 0.914 / 0.382 / 0.363). E1(a) main **+0.485 [+0.382,+0.585]**; replication `rep_named` − `rep_squig30` +0.330 [+0.156,+0.498]; E1(b) filled +0.445 [+0.348,+0.539], line +0.565 [+0.452,+0.662], mean +0.505 [+0.431,+0.572]; regression `named` +2.396 [+1.845,+2.946]. **Grade: Clear.**
  - **Replication gap (said plainly, Step 5 rule):** `rep_named` 0.81 and `rep_squig30` 0.48 vs the paper's 50.6 / 30.5, far above both. The gap between them (+0.33) is larger than the paper's (+0.20). Not explained yet; the likely cause is the attention mask (§8 Step 4 smoke notes: the authors' flash-attention path drops Gemma 3's two-way image attention). Must be stated next to every Gemma number.
  - **E2:** probe layer = stream 2 (raw) and stream 4 (massive-removed, the main view for Gemma, §11). E2(a) at s = 4: unnamed_line 0.955, unnamed_fill 0.830, rep_squig30 0.920 (named 0.985–1.000). **Clear.** E2(b) main +0.487 [+0.382,+0.588], filled +0.453, line +0.569; positive with CI above 0 at all 35 streams in both views. **Clear.** E2(c) main +0.455 [+0.349,+0.557], filled +0.290, line +0.520; positive at 35/35 streams (main) but CI above 0 at only 12/35; filled positive at 23/35. **Moderate** (Qwen: Absent).
  - **E3 at k = 64, S over all streams, massive-removed view** (named_fill / named_line / unnamed_fill / unnamed_line): T1 vocab −0.07 / −0.06 / −0.06 / −0.09; **T2 caption −0.01 / −0.01 / +0.05 / +0.13**; T3 SIEVE −0.10 / −0.10 / −0.23 / −0.29; image-PCA −0.09 / −0.10 / −0.17 / −0.17. Main difference `named_fill` − `unnamed_line`: T1 +0.020 [−0.062,+0.101], **T2 −0.137 [−0.208,−0.075]** (raw −0.230 [−0.367,−0.138]; CI below 0 at 17/35 streams), T3 +0.191 [+0.106,+0.283], image-PCA +0.074 [+0.001,+0.172]. Same reading as Qwen: T2 **Reversed**; T3 points H's way only because the text part is worse than random for unnamed items while S(named) < 0; T1 near 0. **Grade: not supported.**
- **E1 first pass on `qwen` (`python -X utf8 -m nwf.analysis.behaviour --name qwen_fast`, code `src/nwf/analysis/behaviour.py`, output `analysis/e1_qwen_fast.json`):** test items only (ids 100–599; the `rep_` sets are all calibration items, so they are used whole as a replication check), 95% bootstrap CI with 10,000 resamples clustered by the REF shape identity, plus a Logit with cluster-robust errors on the four core sets. Four core accuracies recounted a second way (plain loop): identical. Results: E1(a) `named_fill` − `unnamed_line` = **+0.356 [+0.191,+0.508]**; replication `rep_named` − `rep_squig30` = +0.250 [+0.055,+0.454] (0.58 vs 0.33; paper 54.1 / 29.0); E1(b) named − unnamed within filled +0.230 [+0.056,+0.389], within line +0.338 [+0.181,+0.486], mean +0.284 [+0.168,+0.392]; regression `named` coefficient +1.491 [+0.725,+2.258] (log-odds, with filled, log complexity, lookalike). Accuracies: named_fill 0.678, named_line 0.660, unnamed_fill 0.448, unnamed_line 0.322, named_cx 0.602; squiggle sweep n5 0.542, n10 0.504, n20 0.428, n30 0.322, n50 0.294; taught30 0.332, untaught30 0.326. Base Qwen picks A on only 4–14% of items; letter-balanced accuracy is almost identical. **Grade for E1(a) and E1(b) on Qwen: Clear.** Two choices made by Claude, to confirm with the user: E1(a) main view = `named_fill` vs `unnamed_line` (redrawn shapes, since the `rep_` sets have no test items); bootstrap cluster = REF shape identity (8 clusters in the named sets, so wide CIs). Not yet done: the same for `qwen_names`, the naming-check exclusion (none needed for Qwen: substring matching passes 26/26 in the smoke run), Gemma (waits for its run).
- **E1 on `qwen_names` (`analysis/e1_qwen_names_fast.json`):** name teaching closes the line gap but not the filled one: named_line 0.864 vs unnamed_line 0.866 (diff −0.002 [−0.075,+0.066]); named_fill 0.898 vs unnamed_fill 0.626 (+0.272 [+0.175,+0.365]); squiggle sweep n5 0.818, n10 0.888, n20 0.938, n50 0.636; taught30 0.910, untaught30 0.870; rep 0.88 / 0.94. Context only (E1 gates on base Qwen).
- **E2 on `qwen` (`python -X utf8 -m nwf.analysis.probe --name qwen_fast`, code `src/nwf/analysis/probe.py`, output `analysis/e2_qwen_fast.json`):** probe layer = **stream 3** in both views (pooled calibration curve peaks 0.847 at s=3). Key numbers recomputed by an independent loop script: identical. `rep_squig30` probe curve 0.65 at s=0, lowest 0.37 at s=16, 0.51 at the end: the same shape as the authors' saved file (§4).
  - E2(a): probe at s=3 (test items): unnamed_line 0.736 [0.690,0.778], unnamed_fill 0.760, squig n5 0.932, n10 0.962, n20 0.870, n50 0.512, taught30 0.736, untaught30 0.778; named sets 0.99–1.00. **Clear** (well above 25% everywhere).
  - E2(b) (behaviour among probe-correct items, named − unnamed) at s=3: main `named_fill`−`unnamed_line` **+0.302 [+0.135,+0.458]**; within filled +0.174 [−0.001,+0.336]; within line +0.284 [+0.120,+0.437]; mean +0.229 [+0.110,+0.340]. Positive at all 26 post-injection streams; CI above 0 at 25/26 (main, line) but only 7/26 (filled). Massive-removed view nearly identical. **Grade: Clear for the main view and line, Moderate for filled.**
  - E2(c) (U unnamed − U named) at s=3: main +0.102 [−0.069,+0.259], filled +0.000, line +0.078; but **negative at most other streams** (main positive at 5/26, mean −0.04), because named probe stays at ceiling while unnamed probe falls in the middle layers. **Grade: Absent.** Said plainly: unnamed content is not "more unused" in the U sense; the probe ceiling on named items limits this measure.
  - Not yet done: the linear probe robustness column (design to agree with the user: features, held-out halves, control task).
- **E3 on `qwen` (`python -X utf8 -W ignore -m nwf.analysis.text_share --name qwen_fast`, code `src/nwf/analysis/text_share.py`, output `analysis/e3_qwen_fast.json`):** test items, S averaged over s ≥ 3, cluster bootstrap as E1. Four key S values recomputed by an independent script from the raw npz: identical. §1.3 does not say which of T1/T2/T3 is the main definition; all three are given equal weight (to confirm with the user).
  - **Ceiling:** on named sets random 64-dim subspaces already reach 0.91–1.00 MaxSim accuracy (full ≈ 0.99), so S(named) is pinned near 0 at k = 64 and 256 by construction. Only k = 16 leaves room (random 0.71–0.97).
  - S at k = 64, raw view (named_fill / named_line / unnamed_fill / unnamed_line): **T1 vocab** −0.02 / −0.02 / −0.12 / −0.17; **T2 caption** +0.02 / +0.01 / +0.22 / +0.18; **T3 SIEVE** −0.14 / −0.16 / −0.38 / −0.44; image-PCA (non-text control) −0.03 / −0.02 / −0.25 / −0.23. Massive-removed view: same pattern.
  - Main difference `named_fill`−`unnamed_line` at k = 64: T1 +0.144 [+0.105,+0.190], T2 **−0.155 [−0.228,−0.087]**, T3 +0.296 [+0.221,+0.378], image-PCA +0.201 [+0.160,+0.255]. T2 is negative at all three k (−0.11, −0.15, −0.19) and CI below 0 at 15/26 streams (k = 64).
  - **Reading (said plainly):** (1) T2: squiggle and blob information sits **more** in the caption-text subspace than random subspaces of the same size, and more so than named information: **Reversed** relative to H. (2) T1 and T3 point H's way only because the text part is *worse than random* for unnamed items, while S(named) is ≤ 0, so "S(named) > 0" fails; and the non-text image-PCA control shows the same or a larger effect, so the T1/T3 difference is about high-variance structured directions in general, not about text. (3) S(named) > 0 holds only for T2 and is tiny (+0.01 to +0.06).
  - Also seen: removing the top 64 image-PCA directions raises the squiggle probe from 0.74 to 0.98 at s = 3 (named stays at 1.00): the high-variance directions mask fine shape information. Removing the top-256 caption directions lowers it (0.74 → 0.49 at s = 7).
  - **Grade for E3 on Qwen: not supported (T2 reversed; T1/T3 not text-specific).** Under §1.3 that means **stop or pivot**. Needs a decision by the user (with Maryam and the professor). E4, E5 and the Gemma E3 still to analyse.
  - Subspace sanity on real data (bases.npz): all bases orthonormal (error < 4e-7); top-16 caption directions hold 14–34% of image-token energy (random would hold 0.8%); vocab64 overlaps image-PCA64 and caption64 only at chance level (mean squared cosine 0.03–0.045 vs 0.031).
  - Critique received 2026-11-01 (pasted by the user): E3 is a geometric proxy for a causal claim, and a size-matched random control ignores that the space is anisotropic. Agreed in part; but energy alone does not explain the results either (the highest-energy image-PCA directions do worst on squiggles).
- **E4 on `qwen` (`python -X utf8 -m nwf.analysis.removal --name qwen_fast`, code `src/nwf/analysis/removal.py`, output `analysis/e4_qwen_fast.json`):** items 100–299 of the 4 core sets. Checked: "no removal" logits are bit-identical to the main run on all 800 items; drops recomputed by an independent script: identical.
  - The energy-matched random control removes many directions: for k = 16 between 20 and 1,173 of 2,048 per stream (most 250–900); for k = 2 between 2 and 845.
  - Accuracy (none → caption16 → mean random16): named_fill 0.635 → 0.255 → 0.233; named_line 0.645 → 0.240 → 0.257; unnamed_fill 0.395 → 0.220 → 0.252; unnamed_line 0.335 → 0.220 → 0.263. Every removal pushes accuracy to about chance.
  - **Main view (k = 16): effect −0.040 [−0.107,+0.031]** (filled −0.053, line −0.027). **Grade: Absent.** k = 2 (shown next to it): +0.086 [−0.026,+0.209] (filled +0.017, line +0.155 [+0.017,+0.321]); removing just 2 caption directions drops named accuracy by 0.40–0.47, more than the matched random control (excess +0.15 / +0.21).
  - Floor problem: named sets start about 0.39 above chance, unnamed only 0.08–0.15, so "drops more for named" is partly forced by the floor.
  - **Unnamed accuracy never rises** under any removal, so the Saini "text part masks the residual" story finds no support at the behaviour level either.
- **E5 hint, not a result:** `qwen_names` scores about 0.87 on `untaught30` and 0.91 on `taught30` (all items, calibration included), so teaching names seems to help untaught squiggles too; taught vs untaught may differ little. Report plainly.
- **Open items:** the Gemma replication gap (above); possible 1 GPU h check of Gemma with one-way image attention on the `rep_` sets (needs a small code change; ask the user). Gemma accuracy is also reported as the mean of per-letter accuracies (§11). Naming check uses substring matching (§11). Gemma raw-view probe is dominated by massive activations; use the massive-removed view (§8 Step 4 smoke notes).
- **GPU quota:** 22.85 h used, 7.15 h left (2026-11-01 evening, nothing running); resets **2026-11-07 00:00 UTC (05:00 PKT)**.
- **Next steps:** E5 (`qwen_names` text share on taught30 vs untaught30), the linear-probe robustness column (design to agree), the loud-directions lead, Step 9.

- [x] Step 0 — setup and confirmations
- [x] Step 1 — read and reproduce the authors' pipeline
- [x] Step 2 — stimuli
- [x] Step 3 — pipeline code + CPU tests
- [x] Step 4 — Kaggle smoke + main run (smoke done 2026-10-31; Qwen main run and Gemma reduced run done 2026-11-01)
- [ ] Step 5 — E1 (`qwen` Clear, `qwen_names`, `gemma` Clear, all done 2026-11-01; Gemma replication gap still open)
- [ ] Step 6 — E2 (`qwen` done: Clear, E2(c) Absent; `gemma` done: Clear, E2(c) Moderate; linear probe still to do)
- [ ] Step 7 — E3 (+ E5) (`qwen` and `gemma` E3 done 2026-11-01: not supported in both; E5 still to do)
- [ ] Step 8 — E4 (`qwen` done 2026-11-01: Absent at the main view)
- [ ] Step 9 — write-up and go / no-go
