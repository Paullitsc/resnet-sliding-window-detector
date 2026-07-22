# Presentation Picks — Sections 7 & 8 (Detection & Localization)

These picks come straight from `ECSE415_Project.ipynb` and are meant to fit an **8-minute** presentation. Here is the thing graders care about most: **Method (3 marks)** beats **Results (2)**, then **Discussion (1)**, then **Intro (1)**. So what follows is built around **one slide per big idea**, with the strongest picture you can put on each slide.

Rough target: **about 4 slides** for Part 2, in a full deck of maybe **8–10** slides total.

---

## Slide A — Method: Classifier → Detector Bridge (Section 7)

**Why include it:** Part 2 only works because you made a deliberate choice: **reuse** the ResNet50 you already trained in Part 1 instead of training a whole new detector from scratch. That is exactly the sort of “why did we build it this way?” story the Method rubric (3 marks) is looking for.

**Talking points (≤30 sec) — say it like this:**
- You wrapped the trained ResNet50 in a `DogDetectorWrapper` with one main job: `score_patches()`.
- For each patch you do: `BGR → RGB → Resize to 224×224 → Normalize → softmax → P(dog)`.
- Normalization and architecture match training **exactly**, so later you could swap in another classifier without rewriting the detector logic.

**Visual:** No image needed — you do not need a fancy figure. A short **4-line code snippet** or a tiny flow (`Image → Patches → ResNet → P(dog)`) is plenty. Keep it minimal / simple.

---

## Slide B — Method: Candidate Generation + NMS (Section 7)

**Why include it:** This slide **is** your localization story — it is worth **20 of the 50** Part-2 points. One slide should carry both **how you search** the image and **how you clean up** overlapping guesses.

**Talking points (≤45 sec) — say it like this:**
- You run an **image pyramid** with scale factor **1.25**, stopping when the smallest side hits **128 px**, **stride 32**, and square windows of **`{96, 128, 160, 192}`** pixels (each patch gets resized to **224×224** before the net sees it).
- Windows are **square** on purpose: the classifier learned from **square** crops, so you keep the input “feel” the same as training.
- You keep patches with score **≥ 0.7**, then run **greedy NMS** with **IoU 0.4** so overlapping high-confidence boxes collapse into one.
- Worth one sentence in talk: you process **image-by-image**, **64 patches per batch**, and you **do not** store every patch for the whole dataset — saves memory.

**Visual:** Pick **one** of the **3-panel** figures from `section7_output/` (these are the cell-50 demo outputs). Each panel shows the same image at three stages: *all candidates → above threshold → after NMS*. The Chihuahua `n02085620_275` example is the easiest to read:
- **575** candidates → **432** above threshold → **50** after NMS → best score **1.000**.
- Use `section7_output/fig_002.png` or `section7_output/fig_003.png` — whichever one is the `n02085620_275` Chihuahua run (they were saved in the order cell 50 printed them).

> For the live talk, **drop** the other demo figures. Five sliding-window demos on five different dogs all say the same thing, and you only have eight minutes.

---

## Slide C — Results: IoU Summary + Distribution (Section 8)

**Why include it:** This slide lines up with **Results (2 marks)** and the **Quantitative Evaluation** piece (**10 of 30** Part-2 points). One slide can carry basically your whole numbers story.

**Numbers to put on the slide** (from **cell 55**, **200** Stanford Dogs images):

| Metric | Value |
|---|---|
| Mean IoU | **0.357** |
| Median IoU | 0.319 |
| IoU ≥ 0.5 | **25.5 %** |
| IoU ≥ 0.3 | 52.5 % |
| Images evaluated | 200 |

**Per-breed breakdown** (from **cell 60** — the eval split only has **2** breeds, so this fits on the slide without crowding):

| Breed | n | Mean IoU |
|---|---|---|
| Chihuahua | 152 | 0.355 |
| Japanese spaniel | 48 | 0.363 |

**Visual:** Use the **IoU histogram** from **cell 56** (the plot with the red line at **IoU = 0.5** and orange at **IoU = 0.3**). It is the single best plot for showing *where the detector lives* on the IoU axis — i.e. where most images pile up along IoU. You will need to **re-export** it from the notebook output of cell 56 — it is **not** already saved as a standalone PNG on disk.

**Talking point on honesty** (the rubric explicitly rewards this): something like: *“Mean IoU around 0.36 is not amazing, and we are saying that upfront. The point of the project is to understand **why** we get that number, not to pretend it is state-of-the-art — that is what the next slide is for.”*

---

## Slide D — Results + Discussion: Successes & Failures (Section 8)

**Why include it:** One slide can cover both **Successes & Failures (10 pts)** and **Qualitative Discussion (10 pts)**. Let the pictures do the heavy lifting; your voice explains **why** things worked or broke.

**Recommended layout:** A **2×2 grid** — top row = **2 successes**, bottom row = **2 failures**. If some images are “meh,” leave them off the slide and keep them for **Q&A**.

**Image picks** (from `outputs/qualitative/` — the notebook already saved **10** best/worst cases). Choose the ones where the **prediction vs ground-truth** story is obvious at a glance:
- **Successes** (any **2** from `outputs/qualitative/success/`): `n02085620_2479.jpg`, `n02085620_275.jpg`, `n02085782_1782.jpg`. The same Chihuahua `_275` from Slide B is a nice **callback** if you want a thread through the talk.
- **Failures** (any **2** from `outputs/qualitative/failure/`): `n02085620_10074.jpg` (also used as a stride-test image — probably a **known** hard case), `n02085782_1267.jpg`.

> ⚠ The files in `outputs/qualitative/` are **plain photos** — **not** the version with boxes drawn. Run them through `draw_evaluation_image` (**cell 58**) so **GT (green)** and **prediction (red)** boxes show up; otherwise the audience cannot see what you are comparing.

**Discussion bullets** (verbatim ideas from **cell 61**, condensed):
- **Works when:** the dog is **big**, **roughly centered**, and the **background is simple** — so your sliding-window scales actually line up with the GT box.
- **Fails when:**
  - **Scale** is wrong (dog **tiny** or **huge** compared to your **96–192 px** windows).
  - **Clutter** (grass, furniture, etc.) creates **false positives** that still make it through NMS.
  - **Several dogs** — your top prediction may latch onto the **wrong** dog vs the GT you are grading against.
  - **Weird poses**, **occlusion**, **lying down** — stuff the classifier did not really see in training.
  - **Very long or thin GT boxes** — your windows are **square**, so they never fit well.

Those five bullets map pretty directly onto the rubric question about **lighting, occlusion, background clutter**, and so on.

---

## What to **leave out** of the slides (but be ready for Q&A)

| Cut | Reason |
|---|---|
| The Stanford Dogs ingestion helpers / XML parsing details (cell 44) | Boilerplate plumbing — not really graded, not much insight for the audience. |
| All **5** sliding-window demo images (cell 50 outputs) | **One** panel figure is enough; the rest repeat the same lesson. |
| The random-GT visualization (cell 45 / `fig_000.png`) | Just proves the dataset has dogs; everyone already knows what a dog photo looks like. |
| The full per-image IoU CSV (`outputs/per_image_iou_results.csv`) | Keep it as **backup** if someone asks how per-image IoU was stored or checked. |
| `compute_iou` source code | It is the standard IoU formula — pointing to PyImageSearch (like the project PDF) is fine. |
| Borderline qualitative examples | Fun to discuss, but your success/failure grid already makes the point; only pull these out if asked. |

---

## One-line summary of the Part-2 story arc for the deck

> "We turned a ResNet50 dogs-vs-cats classifier into a detector with sliding windows + NMS, scored 0.36 mean IoU on 200 Stanford Dogs images, and the failures cleanly map onto the assumptions the classifier was trained under (centered, square, single-subject)."
