# Coursework Workflow

## One working copy

Google Drive is the only long-term working directory. GitHub records formal versions, Colab supplies compute, and Codex edits and maintains this same working tree.

- Windows: `G:\我的云端硬盘\umich-dlcv-wi2022`
- Colab: `/content/drive/MyDrive/umich-dlcv-wi2022`
- Account: `dh892127@gmail.com`
- [Drive project folder](https://drive.google.com/drive/folders/17YfzdJD9Ac31gd87U-FNtUDPrjeFeHGy)
- Remote: `https://github.com/Citnarfeng/umich-dlcv-wi2022.git`

The former Downloads repository is a short-term emergency backup. Do not develop or synchronize it independently. The existing `colab/a1/` files are retained historical references to the superseded workflow; the active source and notebooks are in `assignments/a1/`.

## At the start of every task

Run these from the Drive repository:

```sh
git rev-parse --show-toplevel
git status
git branch --show-current
git remote -v
git fetch origin
```

If the working tree is clean and remote changes exist, use `git pull --rebase`. If it is dirty, inspect and protect the work before synchronization; do not reset or overwrite it. Coordinate one writer at a time and wait for Drive sync before switching between Windows and Colab, especially around Git operations.

## Development and Git

Read the task and related code, implement the smallest required change, run the relevant tests, debug, retest, and review `git diff` and `git status`. Preserve course interfaces, tests, notebook cell order, and instructions. Focus on student TODOs. Keep notebook outputs required for review; do not clear them globally.

Commit and push each meaningful verified stage with a factual message. Normal add/commit/push is authorized; no repeated confirmation is needed. Do not force push, rewrite published history, delete branches, reset --hard, or discard uncommitted files without explicit authorization. Store no tokens in remote URLs, code, notebooks, or logs.

## Colab

Open the notebook directly from the Drive project, mount the same account, then enter the assignment directory:

```python
from google.colab import drive
drive.mount('/content/drive')
```

```python
%cd /content/drive/MyDrive/umich-dlcv-wi2022/assignments/a1
```

The A1 setup path is preconfigured. Edit its `.py` modules and `.ipynb` files in Drive, save both, and run the corresponding tests. Use a GPU runtime when required. Do not use “Save a copy in GitHub” as an alternate working-copy mechanism; use the Drive Git working tree for commits and pushes. If GitHub authentication is needed in Colab, use a secure interactive credential flow, never a notebook-embedded token.

Drive browser login, Codex connector authorization, and Colab CLI authentication are separate. The connector account is verified; CLI default credentials were absent at the last check. A desktop Drive mount allows Codex to work without CLI authentication. Colab runtime mounting and full assignment execution still require an actual session.

## Data and results

Keep code on Drive. Large datasets and caches may be staged under `/content`; copy only results worth retaining back to Drive. Do not mistake temporary runtime files for saved code. Keep large checkpoints under `artifacts/checkpoints/` unless assignment requirements call for versioning a specific file. `.pt` and `.pth` are deliberately not ignored globally. Review large files and submission requirements before staging.

## Migration records

The full source repository, including `.git` and uncommitted work, was copied and hash-verified before configuration changes. Original Drive A1 files were retained under `.codex_internal/migration/pre_migration_drive/`. Out-of-repository course resources are preserved under `.codex_internal/migration/source_materials/~other/` for later assignment import, not as a second editable project. A full pre-migration repository archive is also retained. No original local directory was deleted.
