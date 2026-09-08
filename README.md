# Deep Learning for Computer Vision — Coursework

This repository records my programming assignments and experiments while independently studying the University of Michigan's publicly available Deep Learning for Computer Vision materials. I maintain it for review by my own instructor, with implementations, notebook results, and progress tracked through GitHub.

## Course Information

- **Course:** Deep Learning for Computer Vision.
- **Identifier in the supplied assignment materials:** EECS 498-007 / 598-005.
- **Study reference:** Winter 2022.
- **Resources:** [Course website](https://web.eecs.umich.edu/~justincj/teaching/eecs498/WI2022/) · [Schedule](https://web.eecs.umich.edu/~justincj/teaching/eecs498/WI2022/schedule.html).

The starter files retain their original labels and some older setup examples. Their exact correspondence to the Winter 2022 release has not yet been verified.

## Assignments / Progress

Status reflects the currently committed work; preparing a notebook does not mean completing an assignment.

| Assignment | Content | Status | Review location |
| --- | --- | --- | --- |
| Assignment 1 | PyTorch 101; k-Nearest Neighbor | Prepared; implementation pending. TODO placeholders remain; no completed implementation has been verified. | [PyTorch 101](assignments/a1/pytorch101.ipynb) · [k-NN](assignments/a1/knn.ipynb) |
| Assignment 2 | To be added | Not yet added to this repository | — |
| Assignment 3 | To be added | Not yet added to this repository | — |
| Assignment 4 | To be added | Not yet added to this repository | — |
| Assignment 5 | To be added | Not yet added to this repository | — |
| Assignment 6 | To be added | Not yet added to this repository | — |
| Mini-project | To be determined | Not yet added to this repository | — |

**For instructor review:** `assignments/a1/` contains the active implementation files and course notebooks. Progress is based on verified work, not on imported starter outputs. The `colab/a1/` notebooks are historical references from the previous workflow and are no longer development targets.

## Repository Structure

```text
assignments/a1/   # Active .py, .ipynb, and course-provided eecs598 helpers
colab/a1/         # Retained legacy workflow reference; not edited
docs/WORKFLOW.md  # Development, Colab, and synchronization procedure
AGENTS.md        # Project instructions for Codex
```

A2–A6 and the mini-project have not yet been imported into the active assignment tree.

## Workflow

Google Drive holds the single working tree; GitHub records commits; Colab runs the code. [Open the Drive project](https://drive.google.com/drive/folders/17YfzdJD9Ac31gd87U-FNtUDPrjeFeHGy), then open the required notebook in `assignments/a1/` with Colab.

1. Mount Drive and enter `/content/drive/MyDrive/umich-dlcv-wi2022/assignments/a1`.
2. Edit the required `.py` and notebook sections, save both, run tests, and retain relevant results.
3. Review the diff and commit/push meaningful verified stages from the Drive Git working tree.

Large datasets may be staged under `/content`; retain code and final results on Drive. The former local Downloads copy is an emergency backup only. See [the project workflow](docs/WORKFLOW.md) for setup checks and synchronization rules.

## Course-Provided Code

The `eecs598/` package and portions of the starter code, notebook instructions, and test scaffolding come from the original course materials. They are retained because they are needed to run the assignments. **I do not claim these components as my original implementation.** The retained legacy notebook copies have the same origin.

My own work consists of the assignment-required implementations and modifications, written responses, experiments, and resulting notebook outputs as they are completed. Colab setup adaptations support this workflow and do not by themselves constitute completed assignment solutions.

## Environment / Tools

- Python 3, PyTorch, torchvision, and Matplotlib.
- Google Colab for execution and debugging; select a GPU runtime for GPU-dependent exercises.
- Git and GitHub for version control and instructor review.
- CIFAR-10 for the k-NN exercise, downloaded when the relevant cells run.

No pinned environment specification is currently included. Full execution in the current Colab environment has not yet been verified. Files under Colab /content are temporary; code and selected results must be saved to Drive before ending the runtime.

## Disclaimer

I am independently studying publicly available course materials. I am not representing myself as an enrolled University of Michigan student.

This is my personal coursework repository, maintained for review by my own instructor. It is not an official University of Michigan repository or an official solution set. Original course materials and course-provided code are attributed to their respective authors.
