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
| Assignment 1 | PyTorch 101; k-Nearest Neighbor | Prepared; implementation pending. TODO placeholders remain; working notebooks have no saved execution outputs. | [PyTorch 101](colab/a1/pytorch101.ipynb) · [k-NN](colab/a1/knn.ipynb) |
| Assignment 2 | To be added | Not yet added to this repository | — |
| Assignment 3 | To be added | Not yet added to this repository | — |
| Assignment 4 | To be added | Not yet added to this repository | — |
| Assignment 5 | To be added | Not yet added to this repository | — |
| Assignment 6 | To be added | Not yet added to this repository | — |
| Mini-project | To be determined | Not yet added to this repository | — |

**For instructor review:** `colab/a1/` contains the working notebooks, including editable implementations, tests, and space for execution results. `assignments/a1/` is the initial starter snapshot, not evidence of completed work. Progress will be updated as implementations and results are committed.

## Repository Structure

Current tracked structure:

```text
.
├── README.md
├── .gitignore
├── assignments/
│   └── a1/                  # Initial starter snapshot
│       ├── pytorch101.ipynb
│       ├── pytorch101.py
│       ├── knn.ipynb
│       ├── knn.py
│       └── eecs598/          # Course-provided helper package
└── colab/
    └── a1/                  # Working notebooks for implementation and review
        ├── pytorch101.ipynb
        └── knn.ipynb
```

The intended longer-term organization is `assignment1` through `assignment6`, plus `mini_project`. These names are a plan: the repository currently uses the paths above, and later assignment directories have not been created here.

## Workflow

I use Google Colab to run and debug notebooks, and Git commits/pushes to maintain the work on GitHub. The current notebooks also support saving a commit directly from Colab.

| Notebook | Run in Colab | Save path on branch `main` |
| --- | --- | --- |
| PyTorch 101 | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Citnarfeng/umich-dlcv-wi2022/blob/main/colab/a1/pytorch101.ipynb) | `colab/a1/pytorch101.ipynb` |
| k-NN | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Citnarfeng/umich-dlcv-wi2022/blob/main/colab/a1/knn.ipynb) | `colab/a1/knn.ipynb` |

1. Open the latest working notebook. For private-repository access, use the [Colab GitHub browser](https://colab.research.google.com/github), enable **Include private repos**, and authorize the appropriate GitHub account. An instructor also needs repository access to view private files.
2. Run the setup cells in order. Edit the implementation inside the notebook's `%%writefile` cell, rerun that cell, then run the relevant tests. Changes made only to a runtime `.py` file are not saved into the notebook.
3. Keep relevant outputs and written answers. Save each notebook separately using **File → Save a copy in GitHub**, selecting this repository, `main`, and the same path listed above. Confirm the new commit on GitHub; ordinary notebook saving does not confirm a GitHub commit.
4. For local edits, pull the latest changes before editing, then commit and push the updated files. Avoid simultaneous edits to the same notebook in multiple sessions.

The working notebooks embed the required Python modules and recreate them at runtime; setup does not require Drive or cloning the private repository. The starter snapshot does not automatically update with Colab saves. Each notebook includes an optional function to download its current `.py` implementation after the write-file cell has run. If an original-format submission is required, transfer answers to the required template and verify its structure before packaging.

[Google's Colab–GitHub integration guide](https://github.com/googlecolab/colabtools/blob/main/notebooks/colab-github-demo.ipynb)

## Course-Provided Code

The `eecs598/` package and portions of the starter code, notebook instructions, and test scaffolding come from the original course materials. They are retained because they are needed to run the assignments. **I do not claim these components as my original implementation.** Their copies embedded in the Colab notebooks retain the same origin.

My own work consists of the assignment-required implementations and modifications, written responses, experiments, and resulting notebook outputs as they are completed. Colab setup adaptations support this workflow and do not by themselves constitute completed assignment solutions.

## Environment / Tools

- Python 3, PyTorch, torchvision, and Matplotlib.
- Google Colab for execution and debugging; select a GPU runtime for GPU-dependent exercises.
- Git and GitHub for version control and instructor review.
- CIFAR-10 for the k-NN exercise, downloaded when the relevant cells run.

No pinned environment specification is currently included. Full execution in the current Colab environment has not yet been verified. Runtime files and downloaded datasets are temporary and are not preserved by saving a notebook to GitHub.

## Disclaimer

I am independently studying publicly available course materials. I am not representing myself as an enrolled University of Michigan student.

This is my personal coursework repository, maintained for review by my own instructor. It is not an official University of Michigan repository or an official solution set. Original course materials and course-provided code are attributed to their respective authors.
