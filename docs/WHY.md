# Why it's built this way

## The problem in two sentences

Invoice details are still often keyed in by hand, and supplier layouts vary too much for fixed
templates. This project finds where the key fields sit on a page image, so a later step can read
them, and it serves that as a small, tested web service.

## Design choices

**Why detect boxes, instead of running OCR on the whole page or asking a large model?**
Because boxes are cheap, fast and checkable. A small detector runs on a CPU, and anyone can look
at a drawn box and see whether it is on the total. Reading text is a separate step that can be
added per box and measured on its own. Starting with the simplest step that can be measured is
usually better than starting with the most capable model.

**Why YOLOv5, pinned to the v7.0 tag?**
Because the original training recipe used it, and training and serving should run the same
pre- and post-processing code. `torch.hub` fetches code from GitHub, so the tag is pinned rather
than a branch: a branch can change under you between two deployments.

**Why is PyTorch imported in one module only?**
Because most of this code is not machine learning. Validation, batching, thresholds, errors and
the HTTP layer should be testable in seconds on any laptop. The whole suite runs without
PyTorch, weights or a GPU, so CI never downloads several gigabytes of CUDA libraries. The model
sits behind a small `Predictor` interface, and tests replace it with a fake that filters by
threshold the way YOLOv5 does.

**Why one model per process, with a lock around each prediction?**
Because the YOLOv5 wrapper stores its thresholds as attributes on the model object. Two requests
with different thresholds would otherwise overwrite each other. Serialising calls makes the
answers correct at the cost of parallelism. For more throughput, run more processes. That
multiplies memory, which is at least visible and easy to reason about.

**Why do per-request thresholds never touch shared state?**
Because the earlier version wrote them onto the shared model and never reset them. One request
asking for 0.9 silently changed the answer for everyone after it. Overrides are now arguments to
a single call; defaults change only through the admin endpoint.

**Why does the service refuse to start without weights?**
Because the earlier version quietly fell back to generic COCO weights. Those have no invoice
classes, so every answer was empty while every health check said "healthy". A service that
fails loudly at start-up is easier to fix than one that answers with nothing. The fallback still
exists for smoke tests, but it is off by default and labelled.

**Why count ignored detections in every response?**
Because silent dropping hides problems. If the wrong weights are loaded, or boxes fall off the
page, `ignored_detections` goes up and someone can notice. Making uncertainty visible is part of
the output, not an extra.

**Why must a key protect threshold changes and reloads, while reading stays open?**
Because actions that change a running service need stronger controls than actions that only
read from it. Without a key configured, those endpoints are off rather than open.

**Why fingerprint the weights with SHA-256 in every result?**
Because "which model produced this?" should always have an answer. A file name can be reused; a
hash cannot.

**Why decode images with Pillow rather than OpenCV?**
Because YOLOv5 treats NumPy input as RGB and OpenCV reads BGR. The earlier code fed file inputs
to the model with red and blue swapped. Pillow also applies EXIF orientation, so phone photos
arrive the right way up.

**Why does the README publish no accuracy figures?**
Because none can be reproduced from this repository. The earlier README quoted mAP, speed and
cost figures with no dataset, weights or script behind them. A number without a command that
reproduces it is a claim, not a result.

## Questions worth asking

**"How good is it?"**
Unknown, and the documentation says so. To find out: build a labelled test set that no one uses
for training or tuning, describe where it came from, and report per-class precision and recall
from `val.py` with the exact command. Then measure what matters to the people doing the work: are
the extracted field values right, and how much human time does the whole process take, including
checking and correcting? A detector with a high mAP can still save no time if people have to
check every box. Usage is not quality: a tool that people use a lot can still be wrong often.

**"Why not use a document model, such as LayoutLM, Donut or a vision-language model?"**
They may well do better, especially on reading values. They also cost more to run, need more or
different labelled data, and are harder to inspect when they are wrong. The fair way to decide is
to run each option on the same held-out set and compare field-level accuracy, cost per page and
correction time. Using a second model to check the first can catch some errors, but two models
agreeing is not proof that either is right; a human-checked sample still is.

**"Is this safe to run in production?"**
Not as it stands, and the documentation does not claim so. What it has: validated inputs, a
fail-fast start-up, per-request isolation of thresholds, a key on anything that changes the
service, bounded metric labels and logs without file names or client addresses. What it lacks:
authentication on inference, a shared rate limiter, a verified container image, and any evidence
about accuracy. Loading YOLOv5 runs code fetched from GitHub and unpickles the weights file, so
both must be trusted. Being able to build and run the service is not the same as being ready to
deploy it: connecting real invoices, granting access and operating it are separate decisions,
each needing its own checks.

## What's next

1. Generate synthetic, labelled invoice pages so training and evaluation can be reproduced by
   anyone, with no real documents involved.
2. Train a small baseline on that data and publish the `val.py` output with the command that
   produced it.
3. Add a text-reading step per box, and measure end to end: correct field values and total human
   effort.
4. Add CI jobs that load YOLOv5 on CPU and build the Docker image, so the two unverified claims in
   the README become verified or get fixed.
